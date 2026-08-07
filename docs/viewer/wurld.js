/** wurld JS: EBML tag layer, frame records, live recorder + live player (SPEC §9). */

// ---------- EBML ----------
export const ID = {
  EBML_HEADER: 0x1A45DFA3, SEGMENT: 0x18538067, TAGS: 0x1254C367, TAG: 0x7373,
  SIMPLE_TAG: 0x67C8, TAG_NAME: 0x45A3, TAG_STRING: 0x4487, TAG_BINARY: 0x4485,
  CLUSTER: 0x1F43B675, CUES: 0x1C53BB6B,
  SEEK_HEAD: 0x114D9B74, SEEK: 0x4DBB, SEEK_ID: 0x53AB, SEEK_POSITION: 0x53AC,
};

export function readVint(b, pos, keepMarker) {
  const first = b[pos];
  if (!first) throw new Error('bad vint');
  let len = 1; for (let m = 0x80; !(first & m); m >>= 1) len++;
  let val = keepMarker ? first : first & (0xff >> len);
  for (let i = 1; i < len; i++) val = val * 256 + b[pos + i];
  return [val, pos + len, len];
}

export function* ebmlChildren(b, pos, end) {
  while (pos < end) {
    let id, size, sizeLen;
    [id, pos] = readVint(b, pos, true);
    [size, pos, sizeLen] = readVint(b, pos, false);
    const unknown = size === Math.pow(2, 7 * sizeLen) - 1;
    const stop = unknown ? end : pos + size;
    yield [id, pos, stop];
    if (unknown) return;
    pos = stop;
  }
}

function encodeId(id) {
  const n = id > 0xFFFFFF ? 4 : id > 0xFFFF ? 3 : id > 0xFF ? 2 : 1;
  const out = new Uint8Array(n);
  for (let i = n - 1; i >= 0; i--) { out[i] = id & 0xff; id = Math.floor(id / 256); }
  return out;
}

function encodeSize(size) {
  let len = 1;
  while (size >= Math.pow(2, 7 * len) - 1) len++;
  const out = new Uint8Array(len);
  let v = size + Math.pow(2, 7 * len);  // set marker bit
  for (let i = len - 1; i >= 0; i--) { out[i] = v % 256; v = Math.floor(v / 256); }
  return out;
}

export function cat(parts) {
  const total = parts.reduce((s, p) => s + p.length, 0);
  const out = new Uint8Array(total);
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

function el(id, payload) { return cat([encodeId(id), encodeSize(payload.length), payload]); }

/** One Tags element; string values -> TagString, Uint8Array -> TagBinary. */
export function buildTags(entries) {
  const enc = new TextEncoder();
  const tags = [];
  for (const [name, value] of Object.entries(entries)) {
    const valueEl = typeof value === 'string'
      ? el(ID.TAG_STRING, enc.encode(value))
      : el(ID.TAG_BINARY, value);
    tags.push(el(ID.TAG, el(ID.SIMPLE_TAG, cat([el(ID.TAG_NAME, enc.encode(name)), valueEl]))));
  }
  return el(ID.TAGS, cat(tags));
}

/** {segStart, payloadStart} of the Segment (works on a truncated file prefix). */
export function segmentBounds(bytes) {
  let pos = 0;
  while (pos < bytes.length) {
    const segStart = pos;
    let id, size, sizeLen;
    [id, pos] = readVint(bytes, pos, true);
    [size, pos, sizeLen] = readVint(bytes, pos, false);
    if (id === ID.SEGMENT) return { segStart, payloadStart: pos };
    pos += size;
  }
  throw new Error('no Segment element');
}

export function segmentPayloadStart(bytes) { return segmentBounds(bytes).payloadStart; }

function encodeSizeFixed8(size) {
  // BigInt: the 8-byte vint marker (2^56) exceeds Number.MAX_SAFE_INTEGER.
  const out = new Uint8Array(8);
  let v = BigInt(size) | (1n << 56n);
  for (let i = 7; i >= 0; i--) { out[i] = Number(v & 0xFFn); v >>= 8n; }
  return out;
}

/** Standalone file: pre-Segment prefix + Segment(8-byte size, payloadParts). */
export function spliceFile(head, segStart, payloadParts) {
  const payload = cat(payloadParts);
  return cat([head.subarray(0, segStart), encodeId(ID.SEGMENT),
              encodeSizeFixed8(payload.length), payload]);
}

/** Parse a Cues element at the start of buf -> [{timeMs, pos}] (segment-relative). */
export function readCues(buf) {
  let pos = 0;
  let id, size;
  [id, pos] = readVint(buf, pos, true);
  if (id !== ID.CUES) throw new Error('expected Cues element');
  [size, pos] = readVint(buf, pos, false);
  const out = [];
  for (const [cid, cs, ce] of ebmlChildren(buf, pos, pos + size)) {
    if (cid !== 0xBB) continue;  // CuePoint
    let timeMs = null, position = null;
    for (const [fid, fs, fe] of ebmlChildren(buf, cs, ce)) {
      let v = 0;
      if (fid === 0xB3) {  // CueTime
        for (let i = fs; i < fe; i++) v = v * 256 + buf[i];
        timeMs = v;
      } else if (fid === 0xB7) {  // CueTrackPositions
        for (const [gid, gs, ge] of ebmlChildren(buf, fs, fe)) {
          if (gid === 0xF1) {  // CueClusterPosition
            let w = 0;
            for (let i = gs; i < ge; i++) w = w * 256 + buf[i];
            position = w;
          }
        }
      }
    }
    if (timeMs !== null && position !== null) out.push({ timeMs, pos: position });
  }
  return out;
}

/** {elementId: absolute offset} from the SeekHead at the Segment start (SPEC §9.1), or null. */
export function readSeekHead(bytes) {
  const payloadStart = segmentPayloadStart(bytes);
  let pos = payloadStart;
  let id, size, sizeLen;
  [id, pos] = readVint(bytes, pos, true);
  [size, pos, sizeLen] = readVint(bytes, pos, false);
  if (id !== ID.SEEK_HEAD) return null;
  const out = {};
  for (const [sid, ss, se] of ebmlChildren(bytes, pos, pos + size)) {
    if (sid !== ID.SEEK) continue;
    let target = null, position = null;
    for (const [fid, fs, fe] of ebmlChildren(bytes, ss, se)) {
      let v = 0;
      for (let i = fs; i < fe; i++) v = v * 256 + bytes[i];
      if (fid === ID.SEEK_ID) target = v;
      if (fid === ID.SEEK_POSITION) position = v;
    }
    if (target !== null && position !== null) out[target] = payloadStart + position;
  }
  return out;
}

/** All SimpleTags among top-level elements in [start, end) — for header-region prefixes. */
export function collectTagsInRange(bytes, start, end) {
  const out = {};
  for (const [eid, ps, pe] of ebmlChildren(bytes, start, end)) {
    if (eid === ID.TAGS) collectTags(bytes, ps, pe, out);
  }
  return out;
}

/** All SimpleTags in a buffered file: strings last-win, binaries concatenate (SPEC §10). */
export function readWurldTags(bytes) {
  const b = bytes, out = {};
  const dec = new TextDecoder();
  for (const [id, start, end] of ebmlChildren(b, 0, b.length)) {
    if (id !== ID.SEGMENT) continue;
    for (const [cid, cs, ce] of ebmlChildren(b, start, end)) {
      if (cid !== ID.TAGS) continue;
      collectTags(b, cs, ce, out, dec);
    }
  }
  return out;
}

function collectTags(b, start, end, out, dec = new TextDecoder()) {
  for (const [tid, ts, te] of ebmlChildren(b, start, end)) {
    if (tid !== ID.TAG) continue;
    for (const [sid, ss, se] of ebmlChildren(b, ts, te)) {
      if (sid !== ID.SIMPLE_TAG) continue;
      let name = null, value = null;
      for (const [fid, fs, fe] of ebmlChildren(b, ss, se)) {
        if (fid === ID.TAG_NAME) name = dec.decode(b.subarray(fs, fe));
        if (fid === ID.TAG_STRING) value = dec.decode(b.subarray(fs, fe));
        if (fid === ID.TAG_BINARY) value = b.subarray(fs, fe);
      }
      if (name === null || value === null) continue;
      if (value instanceof Uint8Array && out[name] instanceof Uint8Array)
        out[name] = cat([out[name], value]);
      else out[name] = value;
    }
  }
  return out;
}

// ---------- frame records (SPEC §7): 45 bytes LE ----------
export const FRAME_RECORD_SIZE = 45;

export function packFrames(frames, cameraKeys) {
  const camIndex = Object.fromEntries(cameraKeys.map((k, i) => [k, i]));
  const out = new Uint8Array(frames.length * FRAME_RECORD_SIZE);
  const dv = new DataView(out.buffer);
  frames.forEach((f, n) => {
    const o = n * FRAME_RECORD_SIZE;
    dv.setUint32(o, f.i, true);
    dv.setUint32(o + 4, camIndex[f.camera ?? cameraKeys[0]], true);
    dv.setFloat64(o + 8, f.t, true);
    const valid = f.pose_valid !== false;
    const q = valid ? f.q_wxyz : [1, 0, 0, 0], tr = valid ? f.tr : [0, 0, 0];
    q.forEach((v, k) => dv.setFloat32(o + 16 + 4 * k, v, true));
    tr.forEach((v, k) => dv.setFloat32(o + 32 + 4 * k, v, true));
    dv.setUint8(o + 44, valid ? 1 : 0);
  });
  return out;
}

export function unpackFrames(buf, cameraKeys) {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const frames = [];
  for (let o = 0; o + FRAME_RECORD_SIZE <= buf.byteLength; o += FRAME_RECORD_SIZE) {
    const valid = (dv.getUint8(o + 44) & 1) === 1;
    const f = {
      i: dv.getUint32(o, true),
      camera: cameraKeys[dv.getUint32(o + 4, true)],
      t: dv.getFloat64(o + 8, true),
    };
    if (valid) {
      f.q_wxyz = [0, 1, 2, 3].map(k => dv.getFloat32(o + 16 + 4 * k, true));
      f.tr = [0, 1, 2].map(k => dv.getFloat32(o + 32 + 4 * k, true));
    } else f.pose_valid = false;
    frames.push(f);
  }
  return frames;
}

// ---------- incremental top-level splitter ----------
/** Feeds bytes; emits complete top-level elements (descending into the Segment). */
export class StreamSplitter {
  constructor() { this._buf = new Uint8Array(0); this._pos = 0; }

  feed(chunk) {
    this._buf = this._buf.length ? cat([this._buf, chunk]) : new Uint8Array(chunk);
    const out = [];
    for (;;) {
      const got = this._tryElement();
      if (!got) break;
      out.push(got);
      if (this._pos > 1 << 20) { this._buf = this._buf.slice(this._pos); this._pos = 0; }
    }
    return out;
  }

  _tryVint(pos, keepMarker) {
    const b = this._buf;
    if (pos >= b.length) return null;
    if (b[pos] === 0) throw new Error(`invalid EBML vint at ${pos}`);
    let len = 1; for (let m = 0x80; !(b[pos] & m); m >>= 1) len++;
    if (pos + len > b.length) return null;
    return readVint(b, pos, keepMarker);
  }

  _tryElement() {
    const start = this._pos;
    let got = this._tryVint(start, true);
    if (!got) return null;
    const [id, afterId] = got;
    got = this._tryVint(afterId, false);
    if (!got) return null;
    const [size, payloadStart, sizeLen] = got;
    if (size === Math.pow(2, 7 * sizeLen) - 1) {
      if (id !== ID.SEGMENT) throw new Error(`unknown-size element ${id.toString(16)}`);
      // Segment: emit its header bytes, then parse children in place.
      this._pos = payloadStart;
      return { id, bytes: this._buf.slice(start, payloadStart), open: true };
    }
    if (id === ID.SEGMENT) {  // known-size segment (batch file): descend too
      this._pos = payloadStart;
      return { id, bytes: this._buf.slice(start, payloadStart), open: true };
    }
    if (payloadStart + size > this._buf.length) return null;
    this._pos = payloadStart + size;
    return {
      id,
      bytes: this._buf.slice(start, this._pos),
      payload: [payloadStart - start, this._pos - start],  // payload range within bytes
    };
  }
}

// ---------- live recorder ----------
/**
 * Wraps a chromapakz streaming encoder's chunk stream, weaving wurld tags in:
 * header doc after the mux header, WURLD_POSES chunks before each Cluster,
 * a consolidated WURLD_FRAMES on finish. Cues are stripped (their offsets
 * predate the injected tags; SPEC §9 forbids stale Cues).
 */
export class WurldRecorder {
  /**
   * @param makeEncoder  (onChunk) => chromapakz encoder — caller supplies
   *                     createEncoder({...opts, onChunk}) so codec opts stay theirs.
   * @param doc          wurld header document (frames omitted/[]).
   * @param onChunk      receives the wurld-woven byte chunks.
   */
  constructor({ makeEncoder, doc, onChunk }) {
    this._doc = { ...doc, frames: [] };
    this._cameraKeys = Object.keys(doc.cameras ?? {}).sort();
    this._onChunk = onChunk;
    this._pending = [];
    this._all = [];
    this._headerSeen = false;
    this._splitter = new StreamSplitter();
    this._enc = makeEncoder(c => this._weave(c));
  }

  _emit(bytes) { if (bytes.length) this._onChunk(bytes); }

  _flushPoses() {
    if (!this._pending.length) return;
    this._emit(buildTags({ WURLD_POSES: packFrames(this._pending, this._cameraKeys) }));
    this._pending = [];
  }

  _emitHeaderDoc() {
    if (this._headerSeen) return;
    this._headerSeen = true;
    this._emit(buildTags({ WURLD: JSON.stringify(this._doc) }));
  }

  _weave(chunk) {
    for (const elem of this._splitter.feed(chunk)) {
      if (elem.id === ID.CUES) continue;  // stale offsets: drop
      if (elem.id === ID.CLUSTER) { this._emitHeaderDoc(); this._flushPoses(); }
      this._emit(elem.bytes);
      // chromapakz's own Tags is the last header element before clusters.
      if (elem.id === ID.TAGS) this._emitHeaderDoc();
    }
  }

  /** pose: {i, t, camera?, q_wxyz, tr} or {i, t, pose_valid: false}. */
  async addFrame(frame, pose) {
    if (pose) { this._pending.push(pose); this._all.push(pose); }
    await this._enc.addFrame(frame);
  }

  async finish() {
    await this._enc.finish();  // tail cluster(s) arrive via _weave
    this._flushPoses();
    this._emit(buildTags({ WURLD_FRAMES: packFrames(this._all, this._cameraKeys) }));
    return { frames: this._all.length };  // the file itself streamed out via onChunk
  }
}

// ---------- live player ----------
/**
 * Consumes a wurld stream: extracts WURLD* tags into callbacks and
 * forwards everything else (header, chromapakz Tags, Clusters) to `sink`
 * (e.g. a chromapakz network decoder's push()).
 */
export class WurldLivePlayer {
  constructor({ sink, onDoc, onPoses }) {
    this._sink = sink;
    this._onDoc = onDoc;
    this._onPoses = onPoses;
    this._splitter = new StreamSplitter();
    this.doc = null;
    this.frames = [];
    this._cameraKeys = [];
  }

  feed(chunk) {
    for (const elem of this._splitter.feed(chunk)) {
      if (elem.id === ID.TAGS && this._consumeTags(elem)) continue;
      this._sink(elem.bytes);
    }
  }

  _consumeTags(elem) {
    const tags = collectTags(elem.bytes, elem.payload[0], elem.payload[1], {});
    const names = Object.keys(tags);
    if (!names.some(n => n.startsWith('WURLD'))) return false;  // chromapakz's: forward
    const docStr = tags.WURLD;
    if (typeof docStr === 'string') {
      this.doc = JSON.parse(docStr);
      const fb = this.doc.frames_binary;
      this._cameraKeys = fb?.cameras ?? Object.keys(this.doc.cameras ?? {}).sort();
      if (this.doc.frames?.length) {
        this.frames = this.doc.frames.slice();
        this._onPoses?.(this.frames);
      }
      this._onDoc?.(this.doc);
    }
    const posesBuf = tags.WURLD_POSES;
    if (posesBuf instanceof Uint8Array) {
      const chunkFrames = unpackFrames(posesBuf, this._cameraKeys);
      this.frames.push(...chunkFrames);
      this._onPoses?.(chunkFrames);
    }
    const tableBuf = tags.WURLD_FRAMES;
    if (tableBuf instanceof Uint8Array) {
      this.frames = unpackFrames(tableBuf, this._cameraKeys);
      this._onPoses?.(this.frames);
    }
    return true;
  }
}

// Static file server for the viewer tests.
//
// The viewer imports chromapakz from ../node_modules and fetches fixtures from
// conformance/vectors, so the document root has to be the repository root —
// serving viewer/ alone gives a page that cannot load its own decoder.
import { createServer } from 'node:http';
import { createReadStream, statSync } from 'node:fs';
import { extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('../..', import.meta.url));
const PORT = Number(process.env.VIEWER_TEST_PORT || 8791);

const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.webm': 'video/webm',
  '.wasm': 'application/wasm',
};

createServer((req, res) => {
  // Strip the query and refuse to escape the root.
  const rel = normalize(decodeURIComponent(req.url.split('?')[0])).replace(/^(\.\.[/\\])+/, '');
  const path = join(ROOT, rel);
  if (!path.startsWith(ROOT)) { res.writeHead(403).end('no'); return; }
  let st;
  try { st = statSync(path); } catch { res.writeHead(404).end('not found'); return; }
  if (st.isDirectory()) { res.writeHead(404).end('not found'); return; }
  res.writeHead(200, {
    'Content-Type': TYPES[extname(path)] || 'application/octet-stream',
    'Content-Length': st.size,
    // The viewer's remote path issues Range requests; these tests drive the
    // local drop path only, but advertising the capability keeps the page's
    // behaviour the same as in production.
    'Accept-Ranges': 'bytes',
    'Cache-Control': 'no-store',
  });
  createReadStream(path).pipe(res);
}).listen(PORT, '127.0.0.1', () => {
  console.log(`viewer test server on http://127.0.0.1:${PORT}`);
});

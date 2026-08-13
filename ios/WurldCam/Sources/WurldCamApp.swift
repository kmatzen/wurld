import ARKit
import SwiftUI

@main
struct WurldCamApp: App {
    var body: some Scene {
        WindowGroup { ContentView() }
    }
}

struct ContentView: View {
    @StateObject private var capture = CaptureController()
    @State private var sharing = false
    @State private var showingAbout = false
    // First-run walkthrough. The flag persists so it appears exactly once; the
    // ? button reopens it on demand. While the sheet is up the camera has not
    // been started, so the permission prompt arrives after the user has read
    // what the recording is for — not the instant the app opens.
    @AppStorage("hasSeenWelcome") private var hasSeenWelcome = false
    @State private var showingWelcome = false
    @State private var cameraStarted = false

    var body: some View {
        ZStack(alignment: .bottom) {
            #if targetEnvironment(simulator)
            ARPreview(session: capture.session,
                      image: capture.simulated.previewImage).ignoresSafeArea()
            #else
            ARPreview(session: capture.session).ignoresSafeArea()
            #endif
            VStack {
                HStack {
                    Spacer()
                    Button { showingWelcome = true } label: {
                        Image(systemName: "questionmark.circle")
                            .font(.title3)
                            .padding(10)
                            .background(.black.opacity(0.55), in: Circle())
                    }
                    .accessibilityLabel("How to capture")
                    Button { showingAbout = true } label: {
                        Image(systemName: "info.circle")
                            .font(.title3)
                            .padding(10)
                            .background(.black.opacity(0.55), in: Circle())
                    }
                    .accessibilityLabel("About and licenses")
                }
                .padding(.horizontal, 20)
                Spacer()
            }
            .foregroundStyle(.white)
            VStack(spacing: 12) {
                Picker("format", selection: $capture.format) {
                    ForEach(CaptureFormat.allCases) { f in Text(f.rawValue).tag(f) }
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 240)
                .disabled(capture.isRecording)
                // RGB resolution as a fraction of the camera's; depth always
                // records at the native LiDAR grid, whatever is chosen here.
                HStack(spacing: 8) {
                    Text("rgb")
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.7))
                    Picker("rgb resolution", selection: $capture.rgbResolution) {
                        ForEach(RGBResolution.allCases) { r in Text(r.rawValue).tag(r) }
                    }
                    .pickerStyle(.segmented)
                    .frame(maxWidth: 160)
                    .disabled(capture.isRecording)
                }
                .accessibilityLabel("RGB recording resolution")
                Text(capture.statusText)
                    .font(.system(.footnote, design: .monospaced))
                    .padding(6)
                    .background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 8))
                if capture.isRecording {
                    Text("\(capture.frameCount) frames")
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(.red)
                }
                HStack(spacing: 24) {
                    Button {
                        capture.isRecording ? capture.endRecording() : capture.beginRecording()
                    } label: {
                        Circle()
                            .fill(capture.isRecording ? .white : .red)
                            .frame(width: 64, height: 64)
                            .overlay(
                                RoundedRectangle(cornerRadius: capture.isRecording ? 6 : 32)
                                    .fill(.red)
                                    .frame(width: capture.isRecording ? 28 : 56,
                                           height: capture.isRecording ? 28 : 56)
                            )
                    }
                    if capture.lastCaptureURL != nil {
                        Button { sharing = true } label: {
                            Image(systemName: "square.and.arrow.up")
                                .font(.title2)
                                .padding(12)
                                .background(.black.opacity(0.55), in: Circle())
                        }
                    }
                }
            }
            .padding(.bottom, 28)
            .foregroundStyle(.white)
        }
        .onAppear {
            #if targetEnvironment(simulator)
            // Screenshot automation: --simulate-recording drops straight into the
            // recording state so the capture UI can be captured unattended. It
            // must not sit behind the welcome sheet.
            if ProcessInfo.processInfo.arguments.contains("--simulate-recording") {
                startCameraOnce()
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                    capture.beginRecording()
                }
                return
            }
            #endif
            if hasSeenWelcome {
                startCameraOnce()
            } else {
                showingWelcome = true
            }
        }
        .sheet(isPresented: $sharing) {
            if let url = capture.lastCaptureURL {
                ShareSheet(items: [url])
            }
        }
        .sheet(isPresented: $showingAbout) { AcknowledgementsView() }
        .sheet(isPresented: $showingWelcome, onDismiss: {
            // Swiping the sheet away counts as done: the walkthrough must never
            // be able to trap the app. Reopening from ? later must not restart
            // the session, hence the once-guard.
            hasSeenWelcome = true
            startCameraOnce()
        }) {
            WelcomeView { showingWelcome = false }
        }
    }

    private func startCameraOnce() {
        guard !cameraStarted else { return }
        cameraStarted = true
        capture.start()
    }
}

#if targetEnvironment(simulator)
/// Simulator has no camera, so ARSCNView renders nothing. Show the synthetic
/// scene the simulated sensor is feeding the pipeline instead.
struct ARPreview: View {
    let session: ARSession
    let image: UIImage?

    var body: some View {
        GeometryReader { geo in
            if let image {
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    .frame(width: geo.size.width, height: geo.size.height)
                    .clipped()
            } else {
                Color.black
            }
        }
    }
}
#else
struct ARPreview: UIViewRepresentable {
    let session: ARSession

    func makeUIView(context: Context) -> ARSCNView {
        let view = ARSCNView()
        view.session = session
        view.automaticallyUpdatesLighting = false
        return view
    }

    func updateUIView(_ view: ARSCNView, context: Context) {}
}
#endif

struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }

    func updateUIViewController(_ vc: UIActivityViewController, context: Context) {}
}

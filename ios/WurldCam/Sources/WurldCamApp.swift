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

    var body: some View {
        ZStack(alignment: .bottom) {
            ARPreview(session: capture.session).ignoresSafeArea()
            VStack(spacing: 12) {
                Picker("format", selection: $capture.format) {
                    ForEach(CaptureFormat.allCases) { f in Text(f.rawValue).tag(f) }
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 240)
                .disabled(capture.isRecording)
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
        .onAppear { capture.start() }
        .sheet(isPresented: $sharing) {
            if let url = capture.lastCaptureURL {
                ShareSheet(items: [url])
            }
        }
    }
}

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

struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }

    func updateUIViewController(_ vc: UIActivityViewController, context: Context) {}
}

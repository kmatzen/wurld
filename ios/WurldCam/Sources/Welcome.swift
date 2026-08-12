import ARKit
import SwiftUI

/// First-run walkthrough: what the app is for, whether this device can run it,
/// and how to hold it. Three pages, skippable from the first screen, shown once
/// (`@AppStorage` in ContentView) and reopenable from the ? button — the
/// unobtrusive shape of onboarding rather than a gate.
///
/// It also fronts the camera permission: ContentView defers `capture.start()`
/// until this sheet is dismissed, so the system prompt appears after the user
/// has read what the recording is for instead of the instant the app opens.
struct WelcomeView: View {
    var onDone: () -> Void
    // --simulate-no-lidar (simulator QA) opens directly on the device page so
    // the incompatible-device variant can be seen without scripted swiping.
    @State private var page =
        ProcessInfo.processInfo.arguments.contains("--simulate-no-lidar") ? 1 : 0

    /// The one capability the app cannot work without. Checked live via
    /// ARKit — never assumed from the model name — because
    /// `UIRequiredDeviceCapabilities` can only require ARKit in general, so
    /// the App Store will happily install this on a non-Pro iPhone that has
    /// no LiDAR. On devices this is always the real query. The simulator has
    /// no camera, so the real query is always false there; screenshots want
    /// the happy path, hence the default, and `--simulate-no-lidar` drops the
    /// pretence to exercise the incompatible-device page.
    private var hasLiDAR: Bool {
        #if targetEnvironment(simulator)
        !ProcessInfo.processInfo.arguments.contains("--simulate-no-lidar")
        #else
        ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)
        #endif
    }

    var body: some View {
        VStack(spacing: 0) {
            TabView(selection: $page) {
                intent.tag(0)
                device.tag(1)
                howTo.tag(2)
            }
            .tabViewStyle(.page(indexDisplayMode: .always))
            .indexViewStyle(.page(backgroundDisplayMode: .always))

            Button {
                if page < 2 { withAnimation { page += 1 } } else { onDone() }
            } label: {
                Text(page < 2 ? "Continue" : "Start capturing")
                    .font(.headline)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
            }
            .buttonStyle(.borderedProminent)
            .padding(.horizontal, 24)
            .padding(.bottom, 12)

            Button("Skip") { onDone() }
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .padding(.bottom, 16)
                .opacity(page < 2 ? 1 : 0)
        }
        .background(Color(red: 5 / 255, green: 8 / 255, blue: 16 / 255))
        .preferredColorScheme(.dark)
        .interactiveDismissDisabled(false)
    }

    // Copy drafted with an LLM against the app's voice: plain, warm, American
    // English, no numbers-per-second, no formats, no tooling — and truthful
    // about where exploration happens (later, elsewhere), because this app
    // records; it does not itself replay captures from new angles.
    private var intent: some View {
        WelcomePage(
            symbol: "cube.transparent",
            title: "Capture the whole scene",
            lines: [
                "WurldCam records color, depth, and how you move in one take.",
                "Later, explore your moments in 3D on your computer.",
            ])
    }

    private var device: some View {
        WelcomePage(
            symbol: hasLiDAR ? "checkmark.circle" : "exclamationmark.triangle",
            symbolColor: hasLiDAR ? .green : .orange,
            title: hasLiDAR ? "Your camera can do this" : "LiDAR not on this device",
            lines: hasLiDAR
                ? [
                    "LiDAR on this device lets WurldCam capture real depth "
                    + "as you record.",
                    "Later, view your captures in 3D on your computer from "
                    + "new angles.",
                ]
                : [
                    "WurldCam needs LiDAR to capture true depth with video.",
                    "It works on iPhone Pro models since 12 Pro and iPad Pro "
                    + "from 2020.",
                ])
    }

    private var howTo: some View {
        WelcomePage(
            symbol: "figure.walk.motion",
            title: "How to record in 3D",
            lines: [
                "Tap the red button, move slowly around your subject, stay "
                + "1–3 meters away.",
                "Find recordings in Files under WurldCam and share with the "
                + "share button.",
            ])
    }
}

private struct WelcomePage: View {
    let symbol: String
    var symbolColor: Color = .init(red: 255 / 255, green: 191 / 255, blue: 64 / 255)
    let title: String
    let lines: [String]

    var body: some View {
        VStack(spacing: 20) {
            Spacer()
            Image(systemName: symbol)
                .font(.system(size: 56, weight: .light))
                .foregroundStyle(symbolColor)
            Text(title)
                .font(.title2.bold())
                .multilineTextAlignment(.center)
            ForEach(lines, id: \.self) { line in
                Text(line)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer()
            Spacer()
        }
        .padding(.horizontal, 32)
        .foregroundStyle(.white)
    }
}

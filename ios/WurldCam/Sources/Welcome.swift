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
    @State private var page = 0

    /// The one capability the app cannot work without. Checked live rather than
    /// assumed from the model name: `UIRequiredDeviceCapabilities` can only
    /// require ARKit in general, so the App Store will happily install this on
    /// a non-Pro iPhone that has no LiDAR.
    private var hasLiDAR: Bool {
        #if targetEnvironment(simulator)
        true
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

    // Copy drafted with an LLM against the app's voice: plain, warm, no
    // numbers-per-second, no formats, no tooling.
    private var intent: some View {
        WelcomePage(
            symbol: "cube.transparent",
            title: "Capture the whole scene",
            lines: [
                "WurldCam records colour, depth, and how you move in one take.",
                "Revisit your moments later and explore them in 3D.",
            ])
    }

    private var device: some View {
        WelcomePage(
            symbol: hasLiDAR ? "checkmark.circle" : "exclamationmark.triangle",
            symbolColor: hasLiDAR ? .green : .orange,
            title: hasLiDAR ? "Your camera can do this" : "LiDAR not on this device",
            lines: hasLiDAR
                ? [
                    "LiDAR on this device lets WurldCam see real depth as "
                    + "you record.",
                    "Later, walk around your capture, change your view, and "
                    + "see your moment from new angles.",
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
                + "1–3 metres away.",
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

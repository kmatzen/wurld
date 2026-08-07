import SwiftUI

/// Third-party notices. libvpx is BSD-3, whose binary-distribution clause
/// requires the copyright notice, conditions and disclaimer to be reproduced
/// in the materials shipped with the app — so this screen is a licensing
/// obligation, not a courtesy. Keep the texts verbatim.
enum Acknowledgements {
    struct Component: Identifiable {
        let id = UUID()
        let name: String
        let summary: String
        let license: String
    }

    static let components: [Component] = [
        Component(
            name: "libvpx",
            summary: "VP9 encoder — The WebM Project",
            license: """
            Copyright (c) 2010, The WebM Project authors. All rights reserved.

            Redistribution and use in source and binary forms, with or without
            modification, are permitted provided that the following conditions are
            met:

              * Redistributions of source code must retain the above copyright
                notice, this list of conditions and the following disclaimer.

              * Redistributions in binary form must reproduce the above copyright
                notice, this list of conditions and the following disclaimer in
                the documentation and/or other materials provided with the
                distribution.

              * Neither the name of Google, nor the WebM Project, nor the names
                of its contributors may be used to endorse or promote products
                derived from this software without specific prior written
                permission.

            THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
            "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
            LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
            A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
            HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
            SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
            LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
            DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
            THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
            (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
            OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
            """),
        Component(
            name: "ChromaPakZ",
            summary: "Lossless uint16 depth codec",
            license: MIT.text(holder: "Kevin Blackburn-Matzen")),
        Component(
            name: "wurld",
            summary: "Posed sensor-video container format",
            license: MIT.text(holder: "Kevin Blackburn-Matzen")),
    ]

    private enum MIT {
        static func text(holder: String) -> String {
            """
            MIT License

            Copyright (c) 2026 \(holder)

            Permission is hereby granted, free of charge, to any person obtaining a copy
            of this software and associated documentation files (the "Software"), to deal
            in the Software without restriction, including without limitation the rights
            to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
            copies of the Software, and to permit persons to whom the Software is
            furnished to do so, subject to the following conditions:

            The above copyright notice and this permission notice shall be included in all
            copies or substantial portions of the Software.

            THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
            IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
            FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
            AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
            LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
            OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
            SOFTWARE.
            """
        }
    }
}

struct AcknowledgementsView: View {
    @Environment(\.dismiss) private var dismiss

    private var version: String {
        let d = Bundle.main.infoDictionary
        let short = d?["CFBundleShortVersionString"] as? String ?? "?"
        let build = d?["CFBundleVersion"] as? String ?? "?"
        return "\(short) (\(build))"
    }

    var body: some View {
        NavigationStack {
            List {
                Section {
                    LabeledContent("Version", value: version)
                } footer: {
                    Text("Captures are stored on this device only. Nothing is uploaded.")
                }
                ForEach(Acknowledgements.components) { c in
                    Section(c.name) {
                        Text(c.summary).font(.footnote).foregroundStyle(.secondary)
                        Text(c.license)
                            .font(.system(size: 10, design: .monospaced))
                            .textSelection(.enabled)
                    }
                }
            }
            .navigationTitle("About")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .confirmationAction) {
                Button("Done") { dismiss() }
            } }
        }
    }
}

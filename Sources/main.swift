// Twin D500: real-time graphics on both FirePro D500s of a Mac Pro 6,1 at once.
//
//   TwinD500                      open the window
//   TwinD500 --bench <dir>        offscreen benchmark + correctness checks
//   TwinD500 --bench <dir> --only galaxy
//   TwinD500 --selftest <dir>     cycle every scene in the window, save screenshots, quit

import Cocoa

let args = CommandLine.arguments
func argValue(_ flag: String) -> String? {
    guard let i = args.firstIndex(of: flag) else { return nil }
    return i + 1 < args.count ? args[i + 1] : "out"
}

if let dir = argValue("--bench") {
    do { try runBench(outDir: URL(fileURLWithPath: dir), only: argValue("--only")?.lowercased()) } catch {
        print("error: \(error)")
        exit(1)
    }
    exit(0)
}

let delegate: AppDelegate
do {
    delegate = try AppDelegate(gpus: try GPUs(), selftestDir: argValue("--selftest").map { URL(fileURLWithPath: $0) })
} catch {
    print("error: \(error)")
    exit(1)
}
let app = NSApplication.shared
app.delegate = delegate
app.setActivationPolicy(.regular)
app.run()

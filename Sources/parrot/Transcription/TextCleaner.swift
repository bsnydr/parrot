import Foundation
#if canImport(FoundationModels)
import FoundationModels
#endif

/// On-device AI cleanup for dictated text, using Apple's system language model
/// (Foundation Models, macOS 26+ with Apple Intelligence enabled). Removes
/// filler words and false starts, applies self-corrections, and fixes
/// punctuation. Nothing leaves the machine. Any failure or timeout returns the
/// original text unchanged so dictation never breaks.
enum TextCleaner {
    /// Whether cleanup can run right now.
    static var isSupported: Bool { unavailableReason == nil }

    /// A short reason cleanup is unavailable, or nil when it's ready.
    static var unavailableReason: String? {
        #if canImport(FoundationModels)
        if #available(macOS 26, *) {
            switch SystemLanguageModel.default.availability {
            case .available:
                return nil
            case .unavailable(.deviceNotEligible):
                return "this Mac doesn't support Apple Intelligence"
            case .unavailable(.appleIntelligenceNotEnabled):
                return "turn on Apple Intelligence in System Settings"
            case .unavailable(.modelNotReady):
                return "the on-device model is still downloading"
            case .unavailable:
                return "the on-device model is unavailable"
            @unknown default:
                return "the on-device model is unavailable"
            }
        }
        return "requires macOS 26 or later"
        #else
        return "requires macOS 26 or later"
        #endif
    }

    /// Load the language model ahead of time so the first cleanup isn't slow.
    /// No-op when cleanup is unavailable.
    static func prewarm() async {
        #if canImport(FoundationModels)
        if #available(macOS 26, *) {
            guard case .available = SystemLanguageModel.default.availability else { return }
            let session = LanguageModelSession(instructions: instructions)
            _ = try? await session.respond(
                to: "Input: hello there\nOutput:",
                options: GenerationOptions(temperature: 0.2)
            )
        }
        #endif
    }

    /// Clean `text`. Returns the original text on any failure or timeout.
    static func clean(_ text: String) async -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return text }
        #if canImport(FoundationModels)
        if #available(macOS 26, *) {
            guard case .available = SystemLanguageModel.default.availability else { return text }
            let words = trimmed.split(whereSeparator: \.isWhitespace).count
            let session = LanguageModelSession(instructions: instructions)
            // Low temperature keeps the rewrite deterministic; the token cap
            // scales with input so a runaway generation can't spin forever.
            let options = GenerationOptions(
                temperature: 0.2,
                maximumResponseTokens: min(2000, max(128, words * 4))
            )
            // The Input:/Output: framing matches the few-shot examples in the
            // instructions — a bare sentence makes the small model *answer* it
            // instead of cleaning it.
            let prompt = "Input: \(trimmed)\nOutput:"
            // Long dictations legitimately take longer, so the deadline scales
            // with input length.
            let deadline = Double(min(60, 10 + words / 10))
            do {
                let content = try await withTimeout(seconds: deadline) {
                    try await session.respond(to: prompt, options: options).content
                }
                var out = content.trimmingCharacters(in: .whitespacesAndNewlines)
                if out.lowercased().hasPrefix("output:") {
                    out = String(out.dropFirst("Output:".count))
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                }
                return out.isEmpty ? text : out
            } catch {
                return text
            }
        }
        #endif
        return text
    }

    static let instructions = """
    You clean up dictated text.
    - Remove filler words: um, uh, er, ah, and "like" or "you know" when used as filler.
    - Remove false starts and stuttered repeats; keep only the completed thought.
    - When the speaker corrects themselves, keep only the correction.
    - Fix punctuation and capitalization.
    - Never rephrase, summarize, shorten, or add content. Keep the speaker's own words and meaning.
    - Output only the cleaned text, nothing else.

    Examples:
    Input: um so I think we should uh send the report on Tuesday no wait Wednesday
    Output: I think we should send the report on Wednesday.
    Input: can you like grab the the second one from the left
    Output: Can you grab the second one from the left?
    Input: hello there
    Output: Hello there.
    """

    private struct TimedOut: Error {}

    private static func withTimeout<T: Sendable>(
        seconds: Double,
        _ work: @escaping @Sendable () async throws -> T
    ) async throws -> T {
        try await withThrowingTaskGroup(of: T.self) { group in
            group.addTask { try await work() }
            group.addTask {
                try await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
                throw TimedOut()
            }
            let result = try await group.next()!
            group.cancelAll()
            return result
        }
    }
}

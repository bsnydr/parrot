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
            let session = LanguageModelSession(instructions: baseInstructions)
            _ = try? await session.respond(
                to: "Input: hello there\nOutput:",
                options: GenerationOptions(temperature: 0.2)
            )
        }
        #endif
    }

    /// Clean `text`. Returns the original text on any failure or timeout.
    static func clean(_ text: String) async -> String {
        var trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return text }
        let entries = dictionaryEntries()
        trimmed = applyCorrections(trimmed, entries: entries)
        #if canImport(FoundationModels)
        if #available(macOS 26, *) {
            guard case .available = SystemLanguageModel.default.availability else { return trimmed }
            let words = trimmed.split(whereSeparator: \.isWhitespace).count
            let session = LanguageModelSession(
                instructions: instructions(terms: entries.map(\.canonical)))
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
                return out.isEmpty ? trimmed : out
            } catch {
                return trimmed
            }
        }
        #endif
        return trimmed
    }

    /// Personal dictionary: words and names the speaker uses that Whisper tends
    /// to mishear. One entry per line, `#` for comments. Re-read on every
    /// cleanup so edits apply from the next dictation.
    static let dictionaryPath = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent(".config/parrot/dictionary.txt")

    /// Parsed dictionary. A bare `Word` line is a spelling hint for the AI
    /// pass; a `Word: mishear1, mishear2` line also becomes a hard replacement
    /// applied in code before the AI sees the text.
    static func dictionaryEntries() -> [(canonical: String, mishears: [String])] {
        guard let raw = try? String(contentsOf: dictionaryPath, encoding: .utf8) else { return [] }
        return raw.split(separator: "\n").compactMap { line in
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard !trimmed.isEmpty, !trimmed.hasPrefix("#") else { return nil }
            let parts = trimmed.split(separator: ":", maxSplits: 1)
            let canonical = parts[0].trimmingCharacters(in: .whitespaces)
            guard !canonical.isEmpty else { return nil }
            let mishears = parts.count > 1
                ? parts[1].split(separator: ",")
                    .map { $0.trimmingCharacters(in: .whitespaces) }
                    .filter { !$0.isEmpty }
                : []
            return (canonical, mishears)
        }
    }

    /// Case-insensitive whole-word replacement of known mishears.
    /// Deterministic, and works even when the AI model is unavailable.
    static func applyCorrections(
        _ text: String,
        entries: [(canonical: String, mishears: [String])]
    ) -> String {
        var out = text
        for entry in entries {
            for mishear in entry.mishears {
                let pattern = "\\b" + NSRegularExpression.escapedPattern(for: mishear) + "\\b"
                out = out.replacingOccurrences(
                    of: pattern,
                    with: entry.canonical,
                    options: [.regularExpression, .caseInsensitive]
                )
            }
        }
        return out
    }

    static func instructions(terms: [String]) -> String {
        guard !terms.isEmpty else { return baseInstructions }
        return baseInstructions + """


        The speaker uses these special words and names; prefer these exact \
        spellings when a transcribed word sounds like one of them: \
        \(terms.joined(separator: ", ")).
        """
    }

    static let baseInstructions = """
    You clean up dictated text.
    - Remove filler words: um, uh, er, ah, and "like" or "you know" when used as filler.
    - Remove false starts and stuttered repeats; keep only the completed thought.
    - When the speaker corrects themselves, keep only the correction.
    - Fix punctuation and capitalization.
    - Never rephrase, summarize, shorten, or add content. Keep the speaker's own words and meaning.
    - Output plain text only: never markdown, asterisks, or quotation marks around the text.
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

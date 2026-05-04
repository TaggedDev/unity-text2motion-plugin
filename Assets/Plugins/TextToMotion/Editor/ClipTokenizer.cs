// Editor/ClipTokenizer.cs
// Загружает tokenizer.json (HuggingFace формат) и токенизирует строку для CLIP.
// Поддерживает: vocab, merges, BOS/EOS токены.

using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;
using UnityEngine;

namespace TextToMotion.Editor
{
    /// <summary>
    /// Минимальный BPE токенайзер совместимый с CLIP tokenizer.json (HuggingFace).
    /// Загружается из файла один раз, затем используется многократно.
    /// </summary>
    public sealed class ClipTokenizer
    {
        private const int SEQ_LEN = 77;
        private const int SOT     = 49406;   // <|startoftext|>
        private const int EOT     = 49407;   // <|endoftext|>

        private readonly Dictionary<string, int>    _vocab;       // token_str → id
        private readonly Dictionary<(int,int), int> _mergeRanks;  // pair → rank
        private readonly Regex _pat;

        // ── Factory ──────────────────────────────────────────────────────────

        /// <summary>
        /// Загружает tokenizer из tokenizer.json рядом с моделью.
        /// </summary>
        public static ClipTokenizer Load(string tokenizerJsonPath)
        {
            if (!File.Exists(tokenizerJsonPath))
                throw new FileNotFoundException($"tokenizer.json not found: {tokenizerJsonPath}");

            string json = File.ReadAllText(tokenizerJsonPath);
            return new ClipTokenizer(json);
        }

        private ClipTokenizer(string json)
        {
            _vocab      = ParseVocab(json);
            _mergeRanks = ParseMerges(json);

            // CLIP regex: разбивает текст на слова/пунктуацию
            _pat = new Regex(
                @"'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+",
                RegexOptions.Compiled | RegexOptions.IgnoreCase);
        }

        // ── Public API ────────────────────────────────────────────────────────

        /// <summary>
        /// Возвращает int32[77]: [SOT, tokens..., EOT, 0, 0, ...]
        /// </summary>
        public int[] Encode(string text)
        {
            var result = new List<int> { SOT };

            string lower = text.ToLowerInvariant().Trim();
            foreach (Match m in _pat.Matches(lower))
            {
                string word    = m.Value;
                var    encoded = BpeEncode(word + "</w>");
                result.AddRange(encoded);
                if (result.Count >= SEQ_LEN - 1) break;
            }

            // Обрезаем если больше 76 токенов (+ место под EOT)
            if (result.Count > SEQ_LEN - 1)
                result.RemoveRange(SEQ_LEN - 1, result.Count - (SEQ_LEN - 1));

            result.Add(EOT);

            // Padding нулями до 77
            while (result.Count < SEQ_LEN)
                result.Add(0);

            return result.ToArray();
        }

        // ── BPE ───────────────────────────────────────────────────────────────

        private List<int> BpeEncode(string word)
        {
            // Инициализируем символы
            var chars = new List<string>();
            foreach (char c in word)
                chars.Add(c.ToString());

            // BPE merge loop
            while (chars.Count > 1)
            {
                int   bestRank = int.MaxValue;
                int   bestIdx  = -1;

                for (int i = 0; i < chars.Count - 1; i++)
                {
                    if (!_vocab.TryGetValue(chars[i], out int a)) continue;
                    if (!_vocab.TryGetValue(chars[i + 1], out int b)) continue;
                    if (_mergeRanks.TryGetValue((a, b), out int rank) && rank < bestRank)
                    {
                        bestRank = rank;
                        bestIdx  = i;
                    }
                }

                if (bestIdx < 0) break;

                string merged = chars[bestIdx] + chars[bestIdx + 1];
                chars[bestIdx] = merged;
                chars.RemoveAt(bestIdx + 1);
            }

            var ids = new List<int>();
            foreach (string ch in chars)
            {
                if (_vocab.TryGetValue(ch, out int id))
                    ids.Add(id);
                else
                    ids.Add(0); // UNK
            }
            return ids;
        }

        // ── JSON parsing (без внешних библиотек) ─────────────────────────────

        private static Dictionary<string, int> ParseVocab(string json)
        {
            var dict = new Dictionary<string, int>();

            // Ищем блок "vocab": { ... }
            int start = json.IndexOf("\"vocab\"", StringComparison.Ordinal);
            if (start < 0)
            {
                Debug.LogWarning("[TTM] tokenizer.json: 'vocab' section not found");
                return dict;
            }

            int brace = json.IndexOf('{', start);
            if (brace < 0) return dict;

            // Извлекаем содержимое vocab-объекта
            int depth = 0, end = brace;
            for (int i = brace; i < json.Length; i++)
            {
                if (json[i] == '{') depth++;
                else if (json[i] == '}') { depth--; if (depth == 0) { end = i; break; } }
            }

            string vocabBlock = json.Substring(brace + 1, end - brace - 1);

            // Парсим "token": id
            var rx = new Regex(@"""((?:[^""\\]|\\.)*)"":\s*(\d+)");
            foreach (Match m in rx.Matches(vocabBlock))
            {
                string token = Unescape(m.Groups[1].Value);
                int    id    = int.Parse(m.Groups[2].Value);
                dict[token] = id;
            }

            return dict;
        }

        private static Dictionary<(int,int), int> ParseMerges(string json)
        {
            var dict  = new Dictionary<(int,int), int>();
            var vocab = ParseVocab(json);

            // Ищем "merges": [ ... ]
            int start = json.IndexOf("\"merges\"", StringComparison.Ordinal);
            if (start < 0) return dict;

            int bracket = json.IndexOf('[', start);
            if (bracket < 0) return dict;

            int depth = 0, end = bracket;
            for (int i = bracket; i < json.Length; i++)
            {
                if (json[i] == '[') depth++;
                else if (json[i] == ']') { depth--; if (depth == 0) { end = i; break; } }
            }

            string mergesBlock = json.Substring(bracket + 1, end - bracket - 1);

            // Каждый merge — строка вида "aa bb" или ["aa", "bb"]
            // HuggingFace формат: массив строк "left right"
            var rxStr = new Regex(@"""([^""]+)\s+([^""]+)""");
            int rank  = 0;
            foreach (Match m in rxStr.Matches(mergesBlock))
            {
                string left  = m.Groups[1].Value;
                string right = m.Groups[2].Value;
                if (vocab.TryGetValue(left,  out int a) &&
                    vocab.TryGetValue(right, out int b))
                {
                    dict[(a, b)] = rank++;
                }
            }

            return dict;
        }

        private static string Unescape(string s) =>
            s.Replace("\\\"", "\"")
             .Replace("\\\\", "\\")
             .Replace("\\/",  "/")
             .Replace("\\n",  "\n")
             .Replace("\\r",  "\r")
             .Replace("\\t",  "\t");
    }
}
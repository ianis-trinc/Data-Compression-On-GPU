﻿using System.Diagnostics;

namespace LZSS_ST;

class Program
{
    static void Main()
    {
        string repositoryName = "Data-Compression-On-GPU";
        string relativePath = Path.Combine("LZSS-GPU", "test");
        string? basePath = FindRepositoryDirectory(Environment.CurrentDirectory, repositoryName);
        string? folderPath = null;

        if (basePath != null)
        {
            folderPath = Path.Combine(basePath, relativePath);
        }

        // Read input text from file
        if (folderPath != null)
        {
            string inputText = File.ReadAllText(Path.Combine(folderPath, "input.txt"));
            byte[] input = System.Text.Encoding.UTF8.GetBytes(inputText);

            // Measure compression time
            var stopwatch = new Stopwatch();
            stopwatch.Start();
            byte[] compressedData = Compress(input);
            stopwatch.Stop();
            long compressionTime = stopwatch.ElapsedMilliseconds;
            File.WriteAllBytes(Path.Combine(folderPath, "compressed.bin"), compressedData);

            // Measure decompression time
            stopwatch.Restart();
            byte[] decompressedData = Decompress(compressedData);
            stopwatch.Stop();
            long decompressionTime = stopwatch.ElapsedMilliseconds;
            string decompressedText = System.Text.Encoding.UTF8.GetString(decompressedData);
            File.WriteAllText(Path.Combine(folderPath, "decompressed.txt"), decompressedText);

            Console.WriteLine("######################### CPU Compression single thread version #########################\n");
            // Display times
            Console.WriteLine($"Compression time: {compressionTime} ms");
            Console.WriteLine($"Decompression time: {decompressionTime} ms");

            // Check intergrity
            bool isMatch = inputText == decompressedText;
            Console.WriteLine($"Integrity check: {(isMatch ? "PASSED" : "FAILED")}");
        }
    }

    static string? FindRepositoryDirectory(string currentPath, string repositoryName)
    {
        DirectoryInfo? dir = new DirectoryInfo(currentPath);

        while (dir != null)
        {
            if (Directory.Exists(Path.Combine(dir.FullName, repositoryName)))
            {
                return Path.Combine(dir.FullName, repositoryName);
            }
            dir = dir.Parent;
        }

        return null;
    }

    static byte[] Compress(byte[] input)
    {
        int windowSize = 2048;
        int maxMatchLength = 18;
        int inputLength = input.Length;
        var output = new MemoryStream();

        int pos = 0;
        while (pos < inputLength)
        {
            int bestMatchLength = 0;
            int bestMatchDistance = 0;

            for (int j = Math.Max(0, pos - windowSize); j < pos; j++)
            {
                int matchLength = 0;
                while (matchLength < maxMatchLength && pos + matchLength < inputLength &&
                       input[j + matchLength] == input[pos + matchLength])
                {
                    matchLength++;
                }

                if (matchLength > bestMatchLength)
                {
                    bestMatchLength = matchLength;
                    bestMatchDistance = pos - j;
                }
            }

            if (bestMatchLength >= 3)
            {
                output.WriteByte((byte)(1 << 7 | (bestMatchDistance >> 8)));
                output.WriteByte((byte)(bestMatchDistance & 0xFF));
                output.WriteByte((byte)(bestMatchLength - 3));
                pos += bestMatchLength;
            }
            else
            {
                output.WriteByte(0);
                output.WriteByte(input[pos]);
                pos++;
            }
        }

        return output.ToArray();
    }

    static byte[] Decompress(byte[] input)
    {
        var output = new MemoryStream();
        int pos = 0;

        while (pos < input.Length)
        {
            int flag = input[pos] >> 7;
            if (flag == 1)
            {
                int distance = ((input[pos] & 0x7F) << 8) | input[pos + 1];
                int length = input[pos + 2] + 3;
                long start = output.Length - distance;
                for (int i = 0; i < length; i++)
                {
                    output.WriteByte(output.GetBuffer()[start + i]);
                }
                pos += 3;
            }
            else
            {
                output.WriteByte(input[pos + 1]);
                pos += 2;
            }
        }

        return output.ToArray();
    }
}
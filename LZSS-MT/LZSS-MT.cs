using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Collections.Concurrent;
using System.Collections.Generic;

namespace LZSS
{
    class Program
    {
        static void Main(string[] args)
        {
            string folderPath = @"D:\Licenta\LZSS-CS\LZSS-MT\test\";

            // Read input text from file
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

            // Display times
            Console.WriteLine($"Compression time: {compressionTime} ms");
            Console.WriteLine($"Decompression time: {decompressionTime} ms");

            // Integrity check
            bool isMatch = inputText == decompressedText;
            Console.WriteLine($"Integrity check: {(isMatch ? "PASSED" : "FAILED")}");
        }

        static byte[] Compress(byte[] input)
        {
            int numThreads = Environment.ProcessorCount;
            int blockSize = (int)Math.Ceiling(input.Length / (double)numThreads);
            byte[][] compressedBlocks = new byte[numThreads][];
            
            Parallel.For(0, numThreads, i =>
            {
                int start = i * blockSize;
                int length = Math.Min(blockSize, input.Length - start);
                byte[] block = new byte[length];
                Array.Copy(input, start, block, 0, length);
                compressedBlocks[i] = CompressBlock(block);
            });

            return compressedBlocks.SelectMany(block => block).ToArray();
        }

        static byte[] CompressBlock(byte[] input)
        {
            List<byte> compressed = new List<byte>();
            int i = 0;
            int n = input.Length;
            int windowSize = 4096;
            int lookAheadBufferSize = 18;
            int minMatchLength = 3;

            while (i < n)
            {
                int matchLength = 0;
                int matchDistance = 0;

                for (int j = Math.Max(0, i - windowSize); j < i; j++)
                {
                    int k = 0;
                    while (k < lookAheadBufferSize && i + k < n && input[j + k] == input[i + k])
                    {
                        k++;
                    }
                    if (k > matchLength && k >= minMatchLength)
                    {
                        matchLength = k;
                        matchDistance = i - j;
                    }
                }

                if (matchLength >= minMatchLength)
                {
                    compressed.Add(1); // Flag to indicate a match
                    compressed.Add((byte)(matchDistance >> 8));
                    compressed.Add((byte)(matchDistance & 0xFF));
                    compressed.Add((byte)matchLength);
                    i += matchLength;
                }
                else
                {
                    compressed.Add(0); // Flag to indicate a literal
                    compressed.Add(input[i]);
                    i++;
                }
            }

            return compressed.ToArray();
        }

        static byte[] Decompress(byte[] input)
        {
            List<byte> decompressed = new List<byte>();
            int i = 0;

            while (i < input.Length)
            {
                byte flag = input[i++];
                if (flag == 0)
                {
                    if (i < input.Length)
                    {
                        decompressed.Add(input[i++]);
                    }
                }
                else if (flag == 1)
                {
                    if (i + 2 < input.Length)
                    {
                        int matchDistance = (input[i] << 8) | input[i + 1];
                        int matchLength = input[i + 2];
                        i += 3;

                        int start = decompressed.Count - matchDistance;
                        if (start < 0)
                        {
                            throw new Exception("Invalid match distance during decompression");
                        }
                        for (int j = 0; j < matchLength; j++)
                        {
                            decompressed.Add(decompressed[start + j]);
                        }
                    }
                }
                else
                {
                    throw new Exception("Invalid flag value during decompression");
                }
            }

            return decompressed.ToArray();
        }
    }
}

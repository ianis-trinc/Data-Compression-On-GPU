using System;
using System.Diagnostics;
using System.IO;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using System.Text;

namespace LZSS
{
    class Program
    {
        static void Main(string[] args)
        {
            string folderPath = @"D:\Licenta\LZSS-CS\LZSS-GPU\test\";

            // Read input text from file
            string inputText = File.ReadAllText(Path.Combine(folderPath, "input.txt"));
            byte[] input = Encoding.UTF8.GetBytes(inputText);

            // Measure compression time
            var stopwatch = new Stopwatch();
            stopwatch.Start();
            (byte[] compressedData, long gpuCompressionTime) = Compress(input);
            stopwatch.Stop();
            long totalCompressionTime = stopwatch.ElapsedMilliseconds;
            File.WriteAllBytes(Path.Combine(folderPath, "compressed.bin"), compressedData);

            // Measure decompression time
            stopwatch.Restart();
            byte[] decompressedData = Decompress(compressedData);
            stopwatch.Stop();
            long decompressionTime = stopwatch.ElapsedMilliseconds;
            string decompressedText = Encoding.UTF8.GetString(decompressedData);

            try
            {
                File.WriteAllText(Path.Combine(folderPath, "decompressed.txt"), decompressedText);
            }
            catch (IOException ex)
            {
                Console.WriteLine($"Error writing decompressed text to file: {ex.Message}");
            }

            Console.WriteLine("######################### GPU Compression #########################\n");
            // Display times
            Console.WriteLine($"Total compression time (CPU + GPU): {totalCompressionTime} ms");
            Console.WriteLine($"GPU compression time: {gpuCompressionTime} ms");
            Console.WriteLine($"Decompression time: {decompressionTime} ms");

            // Check integrity
            bool isMatch = inputText == decompressedText;
            Console.WriteLine($"Integrity check: {(isMatch ? "PASSED" : "FAILED")}");
            Console.WriteLine($"Original size: {input.Length} bytes");
            Console.WriteLine($"Compressed size: {compressedData.Length} bytes");
            Console.WriteLine($"Compression ratio: {(double)compressedData.Length / input.Length:P2}");
        }

        static (byte[], long) Compress(byte[] input)
        {
            int windowSize = 4096;
            int maxMatchLength = 18;
            int inputLength = input.Length;
            var output = new MemoryStream();
            long gpuCompressionTime = 0;

            // Initialize ILGPU context and accelerator
            using (var context = Context.Create(builder => builder.Cuda()))
            using (var accelerator = context.CreateCudaAccelerator(0))
            {
                // Allocate input and output buffers
                using var inputBuffer = accelerator.Allocate1D(input);
                using var lengthBuffer = accelerator.Allocate1D<int>(inputLength);
                using var distanceBuffer = accelerator.Allocate1D<int>(inputLength);

                // Load and launch kernel
                var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int, int>(FindBestMatchKernel);

                // Measure the GPU kernel execution time
                var gpuStopwatch = new Stopwatch();
                gpuStopwatch.Start();
                kernel(inputLength, inputBuffer.View, lengthBuffer.View, distanceBuffer.View, windowSize, maxMatchLength);
                accelerator.Synchronize();
                gpuStopwatch.Stop();
                gpuCompressionTime = gpuStopwatch.ElapsedMilliseconds;

                // Retrieve the output data
                var lengths = lengthBuffer.GetAsArray1D();
                var distances = distanceBuffer.GetAsArray1D();

                // Process the output data to generate the compressed data
                for (int pos = 0; pos < inputLength;)
                {
                    int bestMatchLength = lengths[pos];
                    int bestMatchDistance = distances[pos];

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
            }

            return (output.ToArray(), gpuCompressionTime);
        }

        static void FindBestMatchKernel(Index1D index, ArrayView<byte> input, ArrayView<int> lengths, ArrayView<int> distances, int windowSize, int maxMatchLength)
        {
            int pos = index;
            if (pos >= input.Length) return;

            int bestMatchLength = 0;
            int bestMatchDistance = 0;

            for (int j = Math.Max(0, pos - windowSize); j < pos; j++)
            {
                int matchLength = 0;
                while (matchLength < maxMatchLength && pos + matchLength < input.Length &&
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

            lengths[pos] = bestMatchLength;
            distances[pos] = bestMatchDistance;
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
                        output.WriteByte((byte)output.GetBuffer()[start + i]);
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
}

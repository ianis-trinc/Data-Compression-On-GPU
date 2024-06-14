using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
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
            string folderPath = @"D:\Licenta\LZSS-CS\LZSS-MT\test\";

            // Read input text from file
            string inputText = File.ReadAllText(Path.Combine(folderPath, "input.txt"));
            byte[] input = Encoding.ASCII.GetBytes(inputText);

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
            string decompressedText = Encoding.ASCII.GetString(decompressedData);
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
            using (var context = Context.Create(builder => builder.Cuda()))
            {
                var cudaDevice = context.GetCudaDevice(0);
                using (var accelerator = cudaDevice.CreateAccelerator(context))
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
                        compressedBlocks[i] = CompressBlock(block, accelerator);
                    });

                    return compressedBlocks.SelectMany(block => block).ToArray();
                }
            }
        }

        static byte[] CompressBlock(byte[] input, Accelerator accelerator)
        {
            using (var stream = accelerator.CreateStream())
            {
                var length = input.Length;
                using (var buffer = accelerator.Allocate1D<byte>(input))
                using (var outputBuffer = accelerator.Allocate1D<byte>(length * 2)) // Assume max double size
                {
                    buffer.CopyFromCPU(input);
                    var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, int>(CompressKernel);
                    kernel(stream, buffer.IntExtent, buffer.View, outputBuffer.View, length);

                    stream.Synchronize();

                    var compressed = outputBuffer.GetAsArray1D();
                    return compressed.Where(b => b != 0).ToArray(); // Remove padding
                }
            }
        }

        static void CompressKernel(Index1D index, ArrayView<byte> input, ArrayView<byte> output, int length)
        {
            int i = index;
            int windowSize = 4096;
            int lookAheadBufferSize = 18;
            int minMatchLength = 3;

            if (i < length)
            {
                int matchLength = 0;
                int matchDistance = 0;

                for (int j = Math.Max(0, i - windowSize); j < i; j++)
                {
                    int k = 0;
                    while (k < lookAheadBufferSize && i + k < length && input[j + k] == input[i + k])
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
                    output[i * 4] = 1; // Flag to indicate a match
                    output[i * 4 + 1] = (byte)(matchDistance >> 8);
                    output[i * 4 + 2] = (byte)(matchDistance & 0xFF);
                    output[i * 4 + 3] = (byte)matchLength;
                }
                else
                {
                    output[i * 2] = 0; // Flag to indicate a literal
                    output[i * 2 + 1] = input[i];
                }
            }
        }

        static byte[] Decompress(byte[] input)
        {
            using (var context = Context.Create(builder => builder.Cuda()))
            {
                var cudaDevice = context.GetCudaDevice(0);
                using (var accelerator = cudaDevice.CreateAccelerator(context))
                {
                    return DecompressBlock(input, accelerator);
                }
            }
        }

        static byte[] DecompressBlock(byte[] input, Accelerator accelerator)
        {
            using (var stream = accelerator.CreateStream())
            {
                var length = input.Length;
                using (var buffer = accelerator.Allocate1D<byte>(input))
                using (var outputBuffer = accelerator.Allocate1D<byte>(length * 2)) // Assume max double size
                {
                    buffer.CopyFromCPU(input);
                    var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, int>(DecompressKernel);
                    kernel(stream, buffer.IntExtent, buffer.View, outputBuffer.View, length);

                    stream.Synchronize();

                    var decompressed = outputBuffer.GetAsArray1D();
                    return decompressed.Where(b => b != 0).ToArray(); // Remove padding
                }
            }
        }

        static void DecompressKernel(Index1D index, ArrayView<byte> input, ArrayView<byte> output, int length)
        {
            int i = index;
            if (i < length)
            {
                byte flag = input[i];
                if (flag == 0)
                {
                    if (i + 1 < length)
                    {
                        output[i] = input[i + 1];
                    }
                }
                else if (flag == 1)
                {
                    if (i + 3 < length)
                    {
                        int matchDistance = (input[i + 1] << 8) | input[i + 2];
                        int matchLength = input[i + 3];

                        int start = i - matchDistance;
                        if (start < 0)
                        {
                            return; // Invalid match distance
                        }
                        for (int j = 0; j < matchLength; j++)
                        {
                            output[i + j] = output[start + j];
                        }
                    }
                }
            }
        }
    }
}

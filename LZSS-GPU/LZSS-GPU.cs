using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace LZSS
{
    class Program
    {
        static void Main(string[] args)
        {
            string folderPath = @"D:\Licenta\LZSS-CS\LZSS-GPU\test\";

            // Read input text from file
            string inputText = File.ReadAllText(Path.Combine(folderPath, "input.txt"));
            byte[] input = System.Text.Encoding.UTF8.GetBytes(inputText);

            // Initialize ILGPU context and CUDA accelerator
            using var context = Context.Create(builder => builder.Cuda());
            using var accelerator = context.CreateCudaAccelerator(0);
            using var stream = accelerator.CreateStream();

            // Measure compression time
            var stopwatch = new Stopwatch();
            stopwatch.Start();
            byte[] compressedData = Compress(input, accelerator, stream);
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

        static byte[] Compress(byte[] input, Accelerator accelerator, AcceleratorStream stream)
        {
            int numThreads = Environment.ProcessorCount;
            int blockSize = (int)Math.Ceiling(input.Length / (double)numThreads);
            int estimatedOutputSize = input.Length / 2; // Estimating output size
            using var inputBuffer = accelerator.Allocate1D<byte>(input.Length);
            using var outputBuffer = accelerator.Allocate1D<byte>(estimatedOutputSize);

            // Copy input data to GPU
            inputBuffer.CopyFromCPU(input);

            // Define a kernel function for compression
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<byte, Stride1D.Dense>, ArrayView1D<byte, Stride1D.Dense>>(CompressKernel);

            // Launch the kernel
            kernel((int)Math.Ceiling(input.Length / 256.0) * 256, inputBuffer.View, outputBuffer.View);

            // Synchronize and copy results back to CPU
            stream.Synchronize();
            byte[] compressedData = new byte[estimatedOutputSize];
            outputBuffer.CopyToCPU(compressedData);

            // Adjust the output size based on the actual compressed data length
            int actualOutputSize = compressedData.Length;
            for (int i = compressedData.Length - 1; i >= 0; i--)
            {
                if (compressedData[i] != 0)
                {
                    actualOutputSize = i + 1;
                    break;
                }
            }
            Array.Resize(ref compressedData, actualOutputSize);

            return compressedData;
        }

        static void CompressKernel(Index1D index, ArrayView1D<byte, Stride1D.Dense> input, ArrayView1D<byte, Stride1D.Dense> output)
        {
            long i = index;
            long n = input.Length;
            int windowSize = 4096;
            int lookAheadBufferSize = 18;
            int minMatchLength = 3;

            if (i < n)
            {
                int matchLength = 0;
                int matchDistance = 0;

                for (long j = Math.Max(0, i - windowSize); j < i; j++)
                {
                    int k = 0;
                    while (k < lookAheadBufferSize && i + k < n && input[j + k] == input[i + k])
                    {
                        k++;
                    }
                    if (k > matchLength && k >= minMatchLength)
                    {
                        matchLength = k;
                        matchDistance = (int)(i - j);
                    }
                }

                if (matchLength >= minMatchLength)
                {
                    if (i * 4 + 3 < output.Length)
                    {
                        output[(int)(i * 4)] = 1; // Flag to indicate a match
                        output[(int)(i * 4 + 1)] = (byte)(matchDistance >> 8);
                        output[(int)(i * 4 + 2)] = (byte)(matchDistance & 0xFF);
                        output[(int)(i * 4 + 3)] = (byte)matchLength;
                    }
                }
                else
                {
                    if (i * 2 + 1 < output.Length)
                    {
                        output[(int)(i * 2)] = 0; // Flag to indicate a literal
                        output[(int)(i * 2 + 1)] = input[i];
                    }
                }
            }
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

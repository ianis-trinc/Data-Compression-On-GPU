using System.Diagnostics;
using System.Text;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace LZSS_GPU;

class Program
{
    static void Main()
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
        int windowSize = 1024;
        int maxMatchLength = 32;
        int inputLength = input.Length;
        var output = new MemoryStream();
        long gpuCompressionTime;

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
                    output.WriteByte(1); // Flag to indicate a match
                    output.WriteByte((byte)(bestMatchDistance >> 8));
                    output.WriteByte((byte)(bestMatchDistance & 0xFF));
                    output.WriteByte((byte)bestMatchLength);
                    pos += bestMatchLength;
                }
                else
                {
                    output.WriteByte(0); // Flag to indicate a literal
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

        int start = Math.Max(0, pos - windowSize);
        int end = (int)Math.Min(input.Length - pos, maxMatchLength);

        for (int j = start; j < pos; j++)
        {
            int matchLength = 0;
            while (matchLength < end && input[j + matchLength] == input[pos + matchLength])
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
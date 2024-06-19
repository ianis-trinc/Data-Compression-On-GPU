#ifdef LZSSCUDA_EXPORTS
#define LZSSCUDA_API __declspec(dllexport)
#else
#define LZSSCUDA_API __declspec(dllimport)
#endif

extern "C" {
    LZSSCUDA_API void CompressCuda(const unsigned char* input, int inputLength, unsigned char* output, int* outputLength);
    LZSSCUDA_API void DecompressCuda(const unsigned char* input, int inputLength, unsigned char* output, int* outputLength);
}

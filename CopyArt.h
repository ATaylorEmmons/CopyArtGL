

struct Image {
   static const int channels = 3;
   int width;
   int height;

   GLuint texture;
   GLenum format;
   void* pixels;

   Image(uint32_t lwidth, uint32_t lheight) {
      width = lwidth;
      height = lheight;
      stbi_set_flip_vertically_on_load(true);
      pixels = (float*)malloc(width*height*channels*sizeof(float));
   }

   Image(std::string path) {
       int channels;
       pixels = (void*)stbi_loadf(path.c_str(), &width, &height, &channels, 3);

       if(!pixels) {
          throw std::invalid_argument("Something was wrong with the image, makes sure the path is correct and that it's format is .png");
       }
   }

   void save(std::string path) {
        if(format != GL_UNSIGNED_BYTE) {
            throw std::invalid_argument("Image format must be in GL_UNSIGNED_BYTE");
        }
          int stride_in_bytes = width*channels*sizeof(uint8_t);
          stbi_write_png(path.c_str(), width, height, channels, (uint8_t*)pixels, stride_in_bytes);
    }


   ~Image() {
      if(pixels) {
          free(pixels);
      }
   }

};

void readFramebuffer(Image& img) {
    img.format = GL_RGB;
    glReadPixels(0, 0, img.width, img.height, GL_RGB, GL_FLOAT, img.pixels);
}

void readFramebufferUByte(Image& img) {
    img.format = GL_UNSIGNED_BYTE;
    glReadPixels(0, 0, img.width, img.height, GL_RGB, GL_UNSIGNED_BYTE, img.pixels);
}

//All in floats
struct MemoryRange {
   uint32_t offset;
   uint32_t length;
};


struct Specimen {

      double score;
      MemoryRange memoryRange;

      Specimen() {}

      Specimen(RNG& rng, MemoryRange memRange, float* memory) {
          memoryRange = memRange;

          for(uint32_t i = memoryRange.offset; i < memoryRange.offset + memoryRange.length; i += 6) {
              memory[i] = rng.runifFloat(-1.0, 1.0);
              memory[i + 1] = rng.runifFloat(-1, 1);

              memory[i + 2] = rng.runifFloat(0, 1);
              memory[i + 3] = rng.runifFloat(0, 1);
              memory[i + 4] = rng.runifFloat(0, 1);
              memory[i + 5] = rng.runifFloat(0, 1);
          }
      }

      Specimen(Specimen& parentA, Specimen& parentB, Specimen& inheritor, RNG& rng, float* memory) {
          memoryRange = inheritor.memoryRange;
          score = 0;

          uint32_t offsetA = parentA.memoryRange.offset;
          uint32_t offsetB = parentB.memoryRange.offset;
          uint32_t inheritOffset = memoryRange.offset;



          int n;
          for(uint32_t i = 0; i < memoryRange.length; i++) {
                n = rng.rbinary();
                memory[inheritOffset + i] = n*memory[offsetA + i] + (1 - n)*memory[offsetB + i] + rng.rnormFloat(0, .001);
          }
      }

      bool operator<(const Specimen &specimen) {
          return score < specimen.score;
      }
};


double fitness(Image& canvas, Image& target) {
    double score = 0;
    uint32_t size = canvas.width*canvas.height*canvas.channels;

    for(uint32_t i = 0; i < size; i++ ) {
        score += abs(((float*)canvas.pixels)[i] - ((float*)target.pixels)[i]);
    }

    return score;
}














//

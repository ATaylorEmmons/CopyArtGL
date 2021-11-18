

struct Image {
   static const int channels = 3;
   int width;
   int height;

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

       stbi_set_flip_vertically_on_load(true);

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
          stbi_flip_vertically_on_write(true);
          stbi_write_png(path.c_str(), width, height, channels, (uint8_t*)pixels, stride_in_bytes);
    }


   ~Image() {
      if(pixels) {
          free(pixels);
      }
   }

};

struct Texture {
    GLuint handle;
    uint32_t width;
    uint32_t height;
    uint32_t unit;

    Texture(uint32_t lwidth, uint32_t lheight, uint32_t unit) {
        width = lwidth;
        height = lheight;

        glGenTextures(1, &handle);
        glActiveTexture(GL_TEXTURE0 + unit);
        glBindTexture(GL_TEXTURE_2D, handle);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, NULL);

    }

    Texture(Image& img, uint32_t unit) {
        width = img.width;
        height = img.height;
        glGenTextures(1, &handle);
        glActiveTexture(GL_TEXTURE0 + unit);

        glBindTexture(GL_TEXTURE_2D, handle);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, img.pixels);
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

void copyFramebuffers(GLuint src, GLuint dest, uint32_t width, uint32_t height) {

    glBindFramebuffer(GL_READ_FRAMEBUFFER, src);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, dest);
    glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_LINEAR);

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

      Specimen(RNG& rng, MemoryRange memRange, float startAlpha, float endAlpha, float* memory) {
          memoryRange = memRange;

          float r;
          float g;
          float b;
          float a;

          float shiftX = 0;
          float shiftY = 0;

          uint32_t count = 0;
          for(uint32_t i = memoryRange.offset; i < memoryRange.offset + memoryRange.length; i += 6) {


              r = rng.runifFloat(0, 1.0);
              g = rng.runifFloat(0, 1.0);
              b = rng.runifFloat(0, 1.0);

              a = rng.runifFloat(endAlpha, startAlpha);

              if(count % 3 == 0) {
                  shiftX = rng.runifFloat(-1.001, .995);
                  shiftY = rng.runifFloat(-1.001, .995);
              }

              memory[i] = rng.runifFloat(.001, .005) + shiftX;
              memory[i + 1] = rng.runifFloat(.001, .005) + shiftY;


              memory[i + 2] = r;
              memory[i + 3] = g;
              memory[i + 4] = b;
              memory[i + 5] = a;

              count++;
          }

      }

      Specimen(Specimen& parentA, Specimen& parentB, Specimen& inheritor, RNG& rng, float mutationChance, float mutationAmount, float* memory) {
          memoryRange = inheritor.memoryRange;
          score = 0;

          uint32_t offsetA = parentA.memoryRange.offset;
          uint32_t offsetB = parentB.memoryRange.offset;

          uint32_t inheritOffset = memoryRange.offset;

          float x;
          float y;

          float r;
          float g;
          float b;
          float a;


          int n;
          int posN;
          uint32_t count = 0;
          for(uint32_t vertex = 0; vertex < memoryRange.length; vertex += 6) {
              int i = vertex;


              if(count % 3 == 0)
                  posN = rng.rbinary();

              x = posN*memory[offsetA + i] + (1 - posN)*memory[offsetB + i] + mutate(rng, mutationChance, mutationAmount);
              y = posN*memory[offsetA + i + 1] + (1 - posN)*memory[offsetB + i + 1] + mutate(rng, mutationChance, mutationAmount);


              n = rng.rbinary();
              r = n*memory[offsetA + i + 2] + (1 - n)*memory[offsetB + i + 2] + mutate(rng, mutationChance, mutationAmount);

              n = rng.rbinary();
              g = n*memory[offsetA + i + 3] + (1 - n)*memory[offsetB + i + 3] + mutate(rng, mutationChance, mutationAmount);

              n = rng.rbinary();
              b = n*memory[offsetA + i + 4] + (1 - n)*memory[offsetB + i + 4] + mutate(rng, mutationChance, mutationAmount);

              n = rng.rbinary();
              a = n*memory[offsetA + i + 5] + (1 - n)*memory[offsetB + i + 5] + mutate(rng, mutationChance, mutationAmount);



              memory[inheritOffset + i] = x;
              memory[inheritOffset + i + 1] = y;

              memory[inheritOffset + i + 2] = r;
              memory[inheritOffset + i + 3] = g;
              memory[inheritOffset + i + 4] = b;
              memory[inheritOffset + i + 5] = a;


              checkBounds(memory[inheritOffset + i]);
              checkBounds(memory[inheritOffset + i + 1]);

              checkBounds(memory[inheritOffset + i + 2]);
              checkBounds(memory[inheritOffset + i + 3]);
              checkBounds(memory[inheritOffset + i + 4]);
              checkBounds(memory[inheritOffset + i + 5]);

              count++;
          }
      }

      float mutate(RNG& rng, float mutationChance, float mutationAmount) {

          float mutate = rng.runifFloat(0.0f, 1.0f);

          if(mutate < mutationChance)
            mutate = 0;
          else
            mutate = 1;

          float amt = rng.runifFloat(-mutationAmount, mutationAmount);//rng.rnormFloat(0, mutationAmount);
          return mutate*amt;
      }

      void checkBounds(float& val) {
        if(val < -1) val = -1;
        if(val > 1) val = 1;
      }

      bool operator<(const Specimen &specimen) {
          return score < specimen.score;
      }
};


double fitness(Image& canvas, Image& target) {
    double score = 0;
    uint32_t size = canvas.width*canvas.height*canvas.channels;

    for(uint32_t i = 0; i < size; i++ ) {
        float a = ((float*)canvas.pixels)[i];
        float b = ((float*)target.pixels)[i];
        score += pow(a - b, 2);
    }

    return score;
}












//

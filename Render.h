struct Texture {
    static const uint32_t channels = 3;

    GLuint handle;
    uint32_t width;
    uint32_t height;
    std::string source;

    Texture(uint32_t width, uint32_t height, std::string source) {
        this->width = width;
        this->height = height;
        this->source = source;

        int l_width = -1;
        int l_height = -1;
        int l_channels = -1;
        stbi_set_flip_vertically_on_load(true);
        float* memory = (float*)stbi_loadf(source.c_str(), &l_width, &l_height, &l_channels, 3);

        if(this->width == -1) {
            throw std::invalid_argument("Something was wrong with the image, makes sure the path is correct and that it's format is .png");
        }

        glGenTextures(1, &handle);
        glBindTexture(GL_TEXTURE_2D, handle);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, this->width, this->height, 0, GL_RGB, GL_FLOAT, memory);


        free(memory);


    }

    Texture(uint32_t width, uint32_t height) {
          this->width = width;
          this->height = height;

          glGenTextures(1, &handle);
          glBindTexture(GL_TEXTURE_2D, handle);

          glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);

      }

      void bind(GLuint unit) {

          glBindImageTexture(unit, handle, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
      }

      void save(std::string path) {

        uint32_t size = width*height*channels*sizeof(uint8_t);
        printf("%d\n", size);
        float* memory = (float*)malloc(size);

        glGetTextureImage(handle, 0, GL_RGB, GL_FLOAT, size, memory);

        int stride_in_bytes = channels*sizeof(uint8_t)*width;

        stbi_flip_vertically_on_write(true);
        stbi_write_png(path.c_str(),
                      width, height, channels, memory, stride_in_bytes);

        free(memory);

    }

};

#include <iostream>
#include <array>
#include <algorithm>

#include "GLAD/glad.c"
#include "GLFW/glfw3.h"

#define STB_ONLY_PNG

#define STB_IMAGE_IMPLEMENTATION
#include "STBI/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "STBI/stb_image_write.h"

#include "Utils.h"

#include "Debug.h"
#include "Shader.h"
#include "Render.h"


struct MemoryReference {
  float* memory;
  uint32_t start;
  uint32_t stop;
};

struct Specimen {
      double score;
      MemoryReference traitMemory;

      bool operator<(const Specimen &specimen) {
          return score < specimen.score;
      }
};

GLFWwindow* initGL(uint32_t width, uint32_t height) {
    if( !glfwInit() )
    {
        printf("Failed to initialize GLFW\n" );
        return nullptr;
    }

    GLFWwindow* window = glfwCreateWindow(width, height, "LearnOpenGL", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return nullptr;
    }
    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        glfwTerminate();
        return nullptr;
    }

    return window;
}


int main() {

    uint32_t width = 1920;
    uint32_t height = 1080;

    std::string targetPath = "lighthouse.png";
    std::string finalImagePath = "out.png";
    std::string frameDirectory = "Frames/";

    const uint32_t generationSize = 1;
    const uint32_t populationSize = 1;
    uint32_t triangleCount = 20;

    RNG rng(Timer::now());

/* Copy Art Declarations*/

    //3*(2 positions + 4 colors)
    uint32_t vertexSize = 18;
    uint32_t traitMemorySize = populationSize*triangleCount*vertexSize*sizeof(float);
    float* traitMemory;
    std::array<Specimen, populationSize> frontPopulation;
    std::array<Specimen, populationSize> backPopulation;

    std::array<Specimen, populationSize>* currentPopulation;
    std::array<Specimen, populationSize>* previousPopulation;
    std::array<Specimen, populationSize>* tempPopulation;


/* Open GL Decalarations*/
    //Must be called before any OpenGl code
    GLFWwindow* window = initGL(width, height);
    Texture target(width, height, targetPath);
    Texture canvas(width, height);

    GLuint shaderProgram = buildAndLinkShaders(vert_Triangle, frag_Triangle);
    GLuint fitnessComputeProgram = buildAndLinkComputeShader(compute_Fitness);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    GLuint VAO;
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    GLuint triangleBuffer;
    glGenBuffers(1, &triangleBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, triangleBuffer);

    int vertexCount = triangleCount*3;
    int sizePerVertex = 6; //2 position, 4 color;

    float* vertices = (float*)malloc(vertexCount*sizePerVertex*sizeof(float));

    for(int i = 0; i < vertexCount*sizePerVertex; i+= sizePerVertex) {
        //Positions
        vertices[i] = rng.runifFloat(-1.0f, 1.0f);
        vertices[i + 1] = rng.runifFloat(-1.0f, 1.0f);

        //Colors
        vertices[i + 2] = rng.runifFloat(0.0f, 1.0f);
        vertices[i + 3] = rng.runifFloat(0.0f, 1.0f);
        vertices[i + 4] = rng.runifFloat(0.0f, 1.0f);
        vertices[i + 5] = rng.runifFloat(0.0f, 1.0f);
    }

    glBufferData(GL_ARRAY_BUFFER, vertexCount*sizePerVertex*sizeof(float), vertices, GL_STREAM_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(2*sizeof(float)));

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);



/* RENDER OFF SCREEN */

    Texture texture(width, height);
    texture.bind(1);

    GLuint framebuffer = 0;
    glGenFramebuffers(1, &framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, texture.handle, 0);
    GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
    glDrawBuffers(1, DrawBuffers);

    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebuffer);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);


/* Shader Storage */

    glUseProgram(fitnessComputeProgram);
    GLuint shaderOutputBuffer = 0;
    glGenBuffers(1, &shaderOutputBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, shaderOutputBuffer);


    glBufferData(GL_SHADER_STORAGE_BUFFER, 1920*1080*sizeof(float), NULL,  GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, shaderOutputBuffer);
    //glShaderStorageBlockBinding(fitnessComputeProgram, 2, 0);


    glDispatchCompute(1920, 1080, 1);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    float val[1920*1080];
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 1920*1080*sizeof(float), &val);

    double score = 0;
    for(int i = 0; i < 1920*1080; i++) {
      score += val[i];
    }
    printf("%f\n", score);


    glUseProgram(shaderProgram);



    while(!glfwWindowShouldClose(window))
    {


        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebuffer);
        glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);


        glDrawArrays(GL_TRIANGLES, 0, vertexCount);

        glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_LINEAR);


        glfwSwapBuffers(window);
        glfwPollEvents();

    }


    glfwTerminate();

    return 0;
}

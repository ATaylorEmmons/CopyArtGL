#include <iostream>
#include <vector>
#include <algorithm>

#include "GLAD/glad.c"
#include "GLFW/glfw3.h"

#define STB_ONLY_PNG

#define STB_IMAGE_IMPLEMENTATION
#include "STBI/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "STBI/stb_image_write.h"

#include "Debug.h"
#include "Utils.h"
#include "Shader.h"
#include "CopyArt.h"

GLFWwindow* initGL(uint32_t width, uint32_t height) {
    if( !glfwInit() )
    {
        printf("Failed to initialize GLFW\n" );
        return nullptr;
    }

    GLFWwindow* window = glfwCreateWindow(width, height, "Copy Art", NULL, NULL);
    if (window == NULL)
    {
        printf("Failed to create GLFW window\n");
        glfwTerminate();
        return nullptr;
    }
    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        printf("Failed to initialize GLAD\n");
        glfwTerminate();
        return nullptr;
    }

    return window;
}

void debug_PrintMem(float* memory, uint32_t size) {

    for(uint32_t i = 0; i < size; i += 6) {
        printf("{%f, %f}, { %f, %f, %f, %f} \n", memory[i], memory[i + 1], memory[i + 2], memory[i + 3], memory[i + 4], memory[i + 5] );
    }

    printf("\n\n\n");
}

__global__ void cuda_fitness(const int N, float* canvasMemory, float* targetMemory, float* storage) {

    __shared__ float cache[1024];

    int t_id = threadIdx.x + blockIdx.x*blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int cacheId = threadIdx.x;


    float store = 0;
    while(t_id < N) {
        store += abs(canvasMemory[t_id] - targetMemory[t_id]);
        t_id += stride;
    }

    cache[cacheId] = store;

    __syncthreads();

    int i = blockDim.x/2;

    while( i != 0) {
      if(cacheId < i) {
        cache[cacheId] += cache[cacheId + i];
      }
      __syncthreads();
      i /= 2;
    }

    if(cacheId == 0) {
      storage[blockIdx.x] = cache[0];
    }
}

int main() {

    uint32_t width = 180;
    uint32_t height = 180;

    std::string targetPath = "Sunset.png";
    std::string finalImagePath = "out.png";
    std::string frameDirectory = "Frames/";

    uint32_t generationCount = 50000;
    uint32_t populationCount = 100;
    uint32_t triangleCount = 500;

    float mutationRate = .001;
    uint32_t eliteCount = 0;
    float selectionCutoff = .25f;

    GLint drawMode = GL_TRIANGLES;//GL_TRIANGLE_STRIP;
    float pointSize = 5; //IF using GL_POINTS

    bool useCuda = true;
    int N = width*height*3;
    int BLOCKS = 1024;
    int THREADS = 1024;


/* INITILIZATION */

    RNG rng(Timer::now());
    Image target(targetPath);
    Image canvas(width, height);

    std::vector<Specimen> frontPopulation;
    std::vector<Specimen> backPopulation;
    frontPopulation.reserve(populationCount);
    backPopulation.reserve(populationCount);

    std::vector<Specimen>* currentPopulation;
    std::vector<Specimen>* previousPopulation;
    std::vector<Specimen>* tempPopPtr;

    uint32_t floatsPerVertex = 6;
    uint32_t floatPerSpecimen = 3*floatsPerVertex*triangleCount;
    uint32_t drawDataMemSize = populationCount*floatPerSpecimen*sizeof(float);

    float* frontLocalBuffer = (float*)malloc(drawDataMemSize);
    float* backLocalBuffer = (float*)malloc(drawDataMemSize);

    float** currentBuffer;
    float** previousBuffer;
    float** tempBuffer;

/* Statistics Variables */
    float lastScore = 0;
    float curScore = 0;

/*OpenGL Init */
    GLFWwindow* window = initGL(width, height);

    GLuint renderProgram = buildAndLinkShaders(vert_Triangle, frag_Triangle);

    glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
    glEnable(GL_BLEND);
    glPointSize(pointSize);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    Texture targetTex(target, 0);
    Texture offscreenTex(width, height, 1);

    GLuint framebuffer;
    glGenFramebuffers(1, &framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, offscreenTex.handle, 0);

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    GLuint deviceBuffer;
    glGenBuffers(1, &deviceBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, deviceBuffer);
    glBufferData(GL_ARRAY_BUFFER, drawDataMemSize, NULL, GL_DYNAMIC_READ);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, floatsPerVertex*sizeof(float), (void*)0);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, floatsPerVertex*sizeof(float), (void*)(2*sizeof(float)));

/*Initilize CUDA */

    uint32_t imageMemorySize = width*height*3*sizeof(float);
    float* deviceTarget;
    float* deviceCanvas;
    float* deviceStorage;
    float* resultStorage;

    if(useCuda) {
        cudaMalloc((void**)&deviceTarget, imageMemorySize);
        cudaMalloc((void**)&deviceCanvas, imageMemorySize);
        cudaMalloc((void**)&deviceStorage, BLOCKS*sizeof(float));
        resultStorage = (float*)malloc(imageMemorySize);

        cudaMemcpy(deviceTarget, (float*)target.pixels, imageMemorySize, cudaMemcpyHostToDevice);

        if(!(deviceTarget && deviceCanvas && deviceStorage && resultStorage)) {
            printf("Failed CUDA intilization.\n");
        }
    }

/* Initilize Population */

    for(uint32_t i = 0; i < populationCount; i++) {
        MemoryRange memRange;
        memRange.offset = i*floatPerSpecimen;
        memRange.length = floatPerSpecimen;

        frontPopulation.push_back(Specimen(rng, memRange, frontLocalBuffer));
        backPopulation.push_back(frontPopulation.at(i));

        memcpy(backLocalBuffer, frontLocalBuffer, drawDataMemSize);
    }

    currentPopulation = &frontPopulation;
    currentBuffer = &frontLocalBuffer;

    previousPopulation = &backPopulation;
    previousBuffer = &frontLocalBuffer;//&backLocalBuffer;

/* Begin Simulation */
    for(uint32_t curGen = 0; curGen < generationCount; curGen++) {

        //Upload generation traits
        glBufferSubData(GL_ARRAY_BUFFER, 0, drawDataMemSize, (*currentBuffer));


        for(uint32_t curSpec = 0; curSpec < populationCount; curSpec++) {

            /* Draw the traits */
            glClear(GL_COLOR_BUFFER_BIT);
            glUseProgram(renderProgram);

            uint32_t start = currentPopulation->at(curSpec).memoryRange.offset/floatsPerVertex;
            uint32_t totalOffset = 3*triangleCount;

            glDrawArrays(drawMode, start, totalOffset);

            readFramebuffer(canvas);

            if(useCuda) {
                cudaMemcpy(deviceCanvas, canvas.pixels, imageMemorySize, cudaMemcpyHostToDevice);
                cuda_fitness <<<BLOCKS, THREADS>>>(N, deviceCanvas, deviceTarget, deviceStorage);
                cudaMemcpy(resultStorage, deviceStorage, BLOCKS*sizeof(float), cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();

                 for(int i = 0; i < BLOCKS; i++) {
                    currentPopulation->at(curSpec).score +=  resultStorage[i];
                 }
            }
            else {

                currentPopulation->at(curSpec).score = fitness(canvas, target);

            }
        }

        std::sort(currentPopulation->begin(), currentPopulation->end());

        lastScore = curScore;
        curScore = currentPopulation->at(0).score;

        float improvement = (lastScore-curScore)/lastScore*100;
        printf("%d, %f, %f\n", curGen, curScore, improvement);

        /* Swap Population Pointers to build current population */
        tempPopPtr = currentPopulation;
        tempBuffer = currentBuffer;

        currentPopulation = previousPopulation;
        currentBuffer = previousBuffer;

        previousPopulation = tempPopPtr;
        previousBuffer = tempBuffer;

        for(uint32_t curElite = 0; curElite < eliteCount; curElite++) {
            currentPopulation->at(curElite) = previousPopulation->at(curElite);
        }

        /* Selection + Crossover */
        for(uint32_t curSpec = eliteCount; curSpec < populationCount; curSpec++) {

            uint32_t indexA = rng.runifInt(0, populationCount*selectionCutoff - 1);
            uint32_t indexB = rng.runifInt(0, populationCount*selectionCutoff - 1);

            while(indexA == indexB) {
                indexB = rng.runifInt(0, populationCount*selectionCutoff - 1);
            }

            Specimen& specA = previousPopulation->at(indexA);
            Specimen& specB = previousPopulation->at(indexB);
            Specimen& inherit = currentPopulation->at(curSpec);

            currentPopulation->at(curSpec) = Specimen(specA, specB, inherit, rng, mutationRate, (*currentBuffer));


        }

        glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_LINEAR);

        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

        glfwPollEvents();
        glfwSwapBuffers(window);

        if(glfwWindowShouldClose(window)) {
            break;
        }
    }


    printf("Completed Simulation.\n");

    while(!glfwWindowShouldClose(window)) {

        glfwPollEvents();
        glfwSwapBuffers(window);
    }


    readFramebufferUByte(canvas);
    canvas.save(finalImagePath);

    glfwTerminate();
    free(frontLocalBuffer);
    free(backLocalBuffer);
    return 0;
}

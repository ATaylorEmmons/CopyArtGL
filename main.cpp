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

int main() {

    uint32_t width = 180;
    uint32_t height = 180;

    std::string targetPath = "Sunset.png";
    std::string finalImagePath = "out.png";
    std::string frameDirectory = "Frames/";

    uint32_t generationCount = 100;
    uint32_t populationCount = 50;
    uint32_t triangleCount = 100;

    uint32_t eliteCount = 0;
    float selectionCutoff = .25f;

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

    GLFWwindow* window = initGL(width, height);

    GLuint renderProgram = buildAndLinkShaders(vert_Triangle, frag_Triangle);
    glUseProgram(renderProgram);

    glClearColor(.5f, .5f, .5f, 1.0f);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

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

        /* Draw the traits */
        for(uint32_t curSpec = 0; curSpec < populationCount; curSpec++) {
            glClear(GL_COLOR_BUFFER_BIT);
            glDrawArrays(GL_TRIANGLES, currentPopulation->at(curSpec).memoryRange.offset/floatsPerVertex, 3*triangleCount);

            readFramebuffer(canvas);
            currentPopulation->at(curSpec).score = fitness(canvas, target);
        }

        std::sort(currentPopulation->begin(), currentPopulation->end());


        printf("%d, %f\n", curGen, currentPopulation->at(0).score);

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

            currentPopulation->at(curSpec) = Specimen(specA, specB, inherit, rng, (*currentBuffer));


        }


        glfwPollEvents();
        glfwSwapBuffers(window);
    }

    printf("Completed Simulation.\n");

    while(!glfwWindowShouldClose(window)) {

        glfwPollEvents();
        glfwSwapBuffers(window);
    }

    glfwTerminate();
    free(frontLocalBuffer);
    free(backLocalBuffer);
    return 0;
}

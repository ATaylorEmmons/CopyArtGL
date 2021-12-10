# CopyArtGL

To build: 
Use nvcc compiler and build main.cu(everything else is pulled in as .h). Link against GLFW and OpenGl. 
GLAD is used to load the modern opengl methods. The one included here is for linux.
the appropriate glad.c and glad.h must be generated from https://glad.dav1d.de/ for the correct platform.

Parameters:
Generation count doesn't need to be above 750 if using enough Epochs.
Population size must be > 8 or the program will crash.





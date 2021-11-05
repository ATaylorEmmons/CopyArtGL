


GLuint buildAndLinkShaders(std::string vertCode, std::string fragCode) {
  	static int count = 0;
  	const char* c_str_vertCode = vertCode.c_str();
  	GLuint vs = glCreateShader(GL_VERTEX_SHADER);
  	glShaderSource(vs, 1, &c_str_vertCode, NULL);
  	glCompileShader(vs);

  	const char* c_str_fragCode = fragCode.c_str();
  	GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
  	glShaderSource(fs, 1, &c_str_fragCode, NULL);
  	glCompileShader(fs);

  	int errorStringSize = 256;
  	std::vector<char> buffer(errorStringSize);
  	glGetShaderiv(vs, GL_INFO_LOG_LENGTH, &errorStringSize);
  	glGetShaderInfoLog(vs, errorStringSize, &errorStringSize, &buffer[0]);

  	debug_printMsg("Vertex Shader error: ");
  	debug_printMsg(std::string(buffer.begin(), buffer.end()));

  	glGetShaderiv(fs, GL_INFO_LOG_LENGTH, &errorStringSize);
  	glGetShaderInfoLog(fs, errorStringSize, &errorStringSize, &buffer[0]);

  	debug_printMsg("Fragment Shader error: ");
  	debug_printMsg(std::string(buffer.begin(), buffer.end()));

  	GLuint shader_program = glCreateProgram();
  	glAttachShader(shader_program, vs);
  	glAttachShader(shader_program, fs);
  	glLinkProgram(shader_program);

  	glDeleteShader(vs);
  	glDeleteShader(fs);

  	count++;
  	return shader_program;
}

GLuint buildAndLinkComputeShader(std::string computeCode) {
    GLuint computeShader = glCreateShader(GL_COMPUTE_SHADER);
    const char* c_str_computeCode = computeCode.c_str();
    glShaderSource(computeShader, 1, &c_str_computeCode, NULL);
    glCompileShader(computeShader);

    GLuint computeProgram = glCreateProgram();
    glAttachShader(computeProgram, computeShader);
    glLinkProgram(computeProgram);

    int errorStringSize = 256;
    std::vector<char> buffer(errorStringSize);
    glGetShaderiv(computeShader, GL_INFO_LOG_LENGTH, &errorStringSize);
    glGetShaderInfoLog(computeShader, errorStringSize, &errorStringSize, &buffer[0]);

    debug_printMsg("Compute Shader error: ");
    debug_printMsg(std::string(buffer.begin(), buffer.end()));


    std::vector<char> programErrorBuffer(errorStringSize);
    glGetProgramiv(computeProgram, GL_INFO_LOG_LENGTH, &errorStringSize);
    glGetProgramInfoLog(computeProgram, errorStringSize, &errorStringSize, &programErrorBuffer[0]);
    debug_printMsg("Compute Program error: ");
    debug_printMsg(std::string(buffer.begin(), buffer.end()));

    return computeProgram;
}

std::string compute_Fitness =
"#version 430\n"
"layout (local_size_x = 1, local_size_y = 1) in;\n"
"layout(rgba32f, binding = 0) uniform image2D img_Target;"
"layout(rgba32f, binding = 1) uniform image2D img_Drawn;"
"layout(std430, binding = 2) buffer Block { float computeOut[1920*1080]; };\n"

"void main() {\n"
"   uint globalId = gl_NumWorkGroups.x*gl_GlobalInvocationID.y + gl_GlobalInvocationID.x;"
"   vec3 target = imageLoad(img_Target, ivec2(gl_LocalInvocationID.x, gl_LocalInvocationID.y )).xyz;"
"   vec3 drawn = imageLoad(img_Drawn, ivec2(gl_LocalInvocationID.x, gl_LocalInvocationID.y )).xyz;"
"   vec3 result = abs(target - drawn);"
"   computeOut[globalId] = dot(result, vec3(1, 1, 1));"
"   barrier();"
"}";

std::string vert_Triangle =
"#version 430\n"
"layout (location = 0) in vec2 attr_Pos;\n"
"layout (location = 1) in vec4 attr_Color;\n"
"out vec4 vertColor;"
"void main() {\n"
"   vertColor = attr_Color;\n"
"   gl_Position = vec4(attr_Pos, 0.0, 1.0);\n"
"}\n";

std::string frag_Triangle =
"#version 430\n"
" in vec4 vertColor;"
" out vec4 frag_color;"
" void main() {\n"
"	frag_color = vertColor;\n "
"}\n";

















//

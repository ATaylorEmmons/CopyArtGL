


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


std::string vert_Triangle =
"#version 450\n"
"layout (location = 0) in vec2 attr_Pos;\n"
"layout (location = 1) in vec4 attr_Color;\n"
"out vec4 vertColor;"
"void main() {\n"
"   vertColor = attr_Color;\n"
"   gl_Position = vec4(attr_Pos, 0.0, 1.0);\n"
"}\n";

std::string frag_Triangle =
"#version 450\n"
" in vec4 vertColor;"
" layout(location = 0) out vec4 frag_color;"
" void main() {\n"
"	frag_color = vertColor;\n "
"}\n";














//

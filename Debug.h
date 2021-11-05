
void debug_printMsg(std::string msg) {
#ifdef DEBUG_BUILD
      std::cout << msg << std::endl;
#endif
}

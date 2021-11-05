
#ifndef __utils_h_
#define __utils_h_


#include <chrono>
#include <random>
#include <iostream>


struct RNG {
    RNG(long randSeed) {

      this->engine = std::default_random_engine{};
      this->engine.seed(randSeed);

    }

    RNG() {}

    ~RNG() {}

   //Includes min and max
   int32_t runifInt(int32_t min, int32_t max) {

        std::uniform_int_distribution<int32_t> randomNumber(min, max);
        //std::default_random_engine generator(rd());

        return randomNumber(engine);

    }


    //Does not include max
    float runifFloat(float min, float max) {

        std::uniform_real_distribution<float> randomNumber(min, max);

        return randomNumber(engine);
    }


    float rnormFloat(float mean, float sd) {

        std::normal_distribution<float> randomNumber(mean, sd);
        //std::default_random_engine generator(rd());

        return randomNumber(engine);

    }

    int rbinary() {
      return rand() % 2;
    }

  private:
    unsigned long seed;
    std::default_random_engine engine;

};



struct Timer {

    std::chrono::high_resolution_clock::time_point t0;
    std::chrono::high_resolution_clock::time_point t1;

    Timer() {}

    static long now() {

      return std::chrono::system_clock::now().time_since_epoch() / std::chrono::milliseconds(1);
    }

    void start() {
        t0 = std::chrono::high_resolution_clock::now();
    }

    long stop() {
        t1 =  std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);

    	  return duration.count();
    }

    long stops() {
      t1 =  std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);

      return duration.count();
    }

    void results(std::string out) {
      std::cout << out << ": " << stop() << std::endl;
    }
};

void println(std::string line) {
  std::cout << line << std::endl;
}

#endif

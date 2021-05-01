#ifndef DATAGENERATOR_H
#define DATAGENERATOR_H

#include <vector>
#include <chrono>

struct TrainingData
{
    std::vector<double> input;
    std::vector<double> target;
};

class DataGenerator
{
public:
    DataGenerator(); // initialize random number generator
    TrainingData generate_AND_data(); // 2 input 1 output
    TrainingData generate_OR_data(); // 2 input 1 output
    TrainingData generate_XOR_data(); // 2 input 1 output
    
    TrainingData generate_Sinus_data(int min, int max); // 1 input 1 output

private:
    double getRandomNum(double max = 1); // maximum value for max = RAND_MAX
};

#endif // !DATAGENERATOR_H
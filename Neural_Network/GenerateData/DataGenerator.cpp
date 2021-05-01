#include "DataGenerator.h"

DataGenerator::DataGenerator()
{
    // initialize std::rand with current time
    std::srand((unsigned int)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
}

TrainingData DataGenerator::generate_AND_data()
{
    double x = 0.0;
    double y = 0.0;
    if (getRandomNum() > 0.5)
    {
        x = 1.0;
    }
    if (getRandomNum() > 0.5)
    {
        y = 1.0;
    }
    if (x == 1.0 && y == 1.0)
        return TrainingData{ {x, y}, {1.0} };
    else
        return TrainingData{ {x, y}, {0.0} };
}

TrainingData DataGenerator::generate_OR_data()
{
    double x = 0.0;
    double y = 0.0;
    if (getRandomNum() > 0.5)
    {
        x = 1.0;
    }
    if (getRandomNum() > 0.5)
    {
        y = 1.0;
    }
    if (x == 1.0 || y == 1.0)
        return TrainingData{ {x, y}, {1.0} };
    else
        return TrainingData{ {x, y}, {0.0} };
}

TrainingData DataGenerator::generate_XOR_data()
{
    double x = 0.0;
    double y = 0.0;
    if (getRandomNum() > 0.5)
    {
        x = 1.0;
    }
    if (getRandomNum() > 0.5)
    {
        y = 1.0;
    }
    if (x != y)
        return TrainingData{ {x, y}, {1.0} };
    else
        return TrainingData{ {x, y}, {0.0} };
}

TrainingData DataGenerator::generate_Sinus_data(int min, int max)
{
    int range = max - min;
    double x = getRandomNum(range) + min;
    return TrainingData{ {x}, {sin(x)} };
}

double DataGenerator::getRandomNum(double max)
{
    return ((double)std::rand() * max) / (double)RAND_MAX;
}

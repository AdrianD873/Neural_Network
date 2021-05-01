#include <iostream>
#include "Network/Network.h"
#include "GenerateData/DataGenerator.h"

const int trainingRounds = 500;

void printInput(std::vector<double> input);
void printTarget(std::vector<double> target);
void printOutput(std::vector<double> output);

int main()
{
    Network network({ 2, 5, 1 });
    DataGenerator generator;

    // train network
    for (int i = 0; i < trainingRounds; i++)
    {
        auto data = generator.generate_OR_data();
        printInput(data.input);
        printTarget(data.target);

        network.feedForward(data.input);
        std::vector<double> output = network.getResults();

        network.backPropagation(data.target);
        network.updateWeights();
        
        printOutput(output);
    }
}

void printInput(std::vector<double> input)
{
    std::cout << "input val: ";
    for (int i = 0; i < input.size(); i++)
    {
        std::cout << input[i] << " ";
    }
    std::cout << "\n";
}

void printTarget(std::vector<double> target)
{
    std::cout << "target val: ";
    for (int i = 0; i < target.size(); i++)
    {
        std::cout << target[i] << " ";
    }
    std::cout << "\n";
}

void printOutput(std::vector<double> output)
{
    std::cout << "output val: ";
    for (int i = 0; i < output.size(); i++)
    {
        std::cout << output[i] << " ";
    }
    std::cout << "\n\n";
}

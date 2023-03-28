#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <random>

class MLP{
    private:
        int dimension;
        int units;
        float learningRate; 
        float gradients;
        float loss;
        std::vector<std::vector<double>> weightsTransposed;

        void initialiseWeightsWithHe();
        double dotProduct(std::vector<double> vectorOne, std::vector<double> vectorTwo);
        double relu(double output);
        std::vector<double> feedForward(std::vector<std::vector<double>> input);
        float rootMeanSquaredLoss (std::vector<double>groundTruth, std::vector<double> predictions);
        void backpropagate();

    public:
        MLP(int nUnits, int nFeatures,float gamma);
        void printWeights();
};



MLP::MLP(int nUnits, int nFeatures, float gamma){
    dimension = nFeatures + 1;
    units = nUnits;
    learningRate = gamma;
    initialiseWeightsWithHe();
}


void MLP::initialiseWeightsWithHe(){
    const double standardDeviation = std::sqrt(2.0 / dimension);
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, standardDeviation);

    for (int feature = 0; feature < dimension; feature++){
        std::vector<double> row(units, 0.0);
        for (int node = 0; node < units; node++){
            row[node] = distribution(generator);
        }
        weightsTransposed.push_back(row);
    }
}

void MLP::printWeights(){
    for (int feature = 0; feature < dimension; feature++){
        for (int node = 0; node < units; node++){
            std::cout << weightsTransposed[feature][node] << " ";
        }
        std::cout << std::endl;
    }
}


double MLP::relu(double output){
    return (output > 0.0) ? output : 0.0;
}


double MLP::dotProduct(std::vector<double> vectorOne, std::vector<double> vectorTwo){
    double result = 0.0;
    for (int idx=0; idx < vectorOne.size(); idx++){
        result += (vectorOne[idx] * vectorTwo[idx]);
    }
    return result;
}


std::vector<double> MLP::feedForward(std::vector<std::vector<double>> input){
    
}


float MLP::rootMeanSquaredLoss (std::vector<double>groundTruth, std::vector<double> predictions){

}
        
        
void MLP::backpropagate(){

}




int main(){

    MLP model(3, 100, 0.001);
    model.printWeights();
    return 0;
}

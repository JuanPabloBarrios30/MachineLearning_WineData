#include "exeigennorm.h"
#include "linealregression.h"

#include <iostream>
#include <stdlib.h>
#include <Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include <vector>

/* En primer lugar se creara una clase llamada "ExEigenNorm", la cual nos permitira
 * leer un dataset, extraer los datos, montar sobre la estructura Eigen para normalizar los datos.
 */

int main(int argc, char *argv[])
{
    /* Se crea un objeto del tipo ExEigenNorm, se incluye los tres argumentos del constructor:
     * 1. Nombre del dataset
     * 2. Delimitador
     * 3. Flag (header o no)*/
    ExEigenNorm extraccion(argv[1] , argv [2], argv [3]);
    LinealRegression LR;

    /*Se leen los datos del fichero por la funcion LeerCSV()*/
    std::vector<std::vector<std::string>> dataFrame = extraccion.LeerCSV();
    /*
     * Para probar la segunda funcion seria CSVtoEigen() se define la cantidad de filas y columnas
     * basados en los datos de entrada
     */
    int filas = dataFrame.size();
    int columnas = dataFrame[0].size();
    Eigen::MatrixXd matrizDataF = extraccion.CSVtoEigen(dataFrame,filas,columnas);

    //std::cout<<matrizDataF<<std::endl;

    /* Para desarrolar el primer algoritmo de regresion lineal, en donde se probará con los datos de los vinos (winedata.csv)
     * se presentará la regresión lineal para multiples variables. Dada la naturaleza de la regresión lineal, si se tiene variables con
     * diferentes unidades, una variable podria beneficiar/estropear otra variable: se necesitará estandarizar los datos,
     * dejando a todas las variables del mismo orden de magnitud y centradas en cero. Para ello se construirá una función de
     * normalización basada en el set score normalización. Se necesitan tres funciones: la funcion de normalizacion,
     * la del promedio y la desviacion estandar.*/

    /* Se muestran los datos normalizados*/

    Eigen::MatrixXd normMatriz = extraccion.Normalizacion(matrizDataF);
   // std::cout<<"Datos normalizados:\n"<<std::endl;
    //std::cout << normMatriz << std::endl;

    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> divDatos = extraccion.TrainTestSplit(normMatriz, 0.8);
    /*Se desempaca la tupla, se usa std::tie
     * https://en.cppreference.com/w/cpp/utility/tuple/tie */
    Eigen::MatrixXd X_Train, y_Train, X_Test, y_Test;

    std::tie(X_Train, y_Train, X_Test, y_Test) = divDatos;

    /*Inspeccion visual de la division de los datos pata entrenamiento y prueba

    std::cout << "\n\nTamaño original:                 ->" << normMatriz.rows() << std::endl;
    std::cout << "Tamaño entrenamiento (filas):    ->" << X_Train.rows() << std::endl;
    std::cout << "Tamaño entrenamiento (columnas): -> " << X_Train.cols() << std::endl;
    std::cout << "Tamaño prueba (filas):           ->" << X_Test.rows() << std::endl;
    std::cout << "Tamaño prueba (columnas):        ->" << X_Test.cols() << std::endl;

    std::cout << "\n\nTamaño original:                 ->" << normMatriz.rows() << std::endl;
    std::cout << "Tamaño entrenamiento (filas):    ->" << y_Train.rows() << std::endl;
    std::cout << "Tamaño entrenamiento (columnas): ->" << y_Train.cols() << std::endl;
    std::cout << "Tamaño prueba (filas):           ->" << y_Test.rows() << std::endl;
    std::cout << "Tamaño prueba (columnas):        ->" << y_Test.cols() << std::endl;*/



    /* A continuacion se procede a probar la clase de regresion lineal */

    Eigen::VectorXd vectorTrain = Eigen::VectorXd::Ones(X_Train.rows());
    Eigen::VectorXd vectorTest = Eigen::VectorXd::Ones(X_Test.rows());

    /* Redimension de las matrices para ubicacion en los vectores de ONES (similar a reshape
     * con Numpy) */

    X_Train.conservativeResize(X_Train.rows(), X_Train.cols() + 1);
    X_Train.col(X_Train.cols() - 1) = vectorTrain;

    X_Test.conservativeResize(X_Test.rows(), X_Test.cols() + 1);
    X_Test.col(X_Test.cols() - 1) = vectorTest;

    /* Se define el vector theta que pasara al algoritmo de gradiente descendiente (basicamente
     * un vector de ZEROS del mismo tamaño del vector de entrenamiento. Adicional se pasara
     * alpha y el numero de iteraciones*/

    Eigen::VectorXd  theta = Eigen::VectorXd::Zero(X_Train.cols());
    float alpha = 0.01;
    int iteraciones = 1000;

    /* Se definen las variables de salida que representan los coeficientes y el vector de costo*/
    Eigen::VectorXd thetaOut;
    std::vector<float> costo;

    std::tuple<Eigen::VectorXd, std::vector<float>> gradienteD = LR.GradienteDescendiente(X_Train, y_Train, theta, alpha, iteraciones);
    std::tie(thetaOut, costo) = gradienteD;

    /*Se imprimen los valores de los coeficientes theta para cada FEATURES*/

    //std::cout<<"\nTheta: \n" << thetaOut <<std::endl;
    std::cout<<"\nCosto: \n"<<std::endl;
    for(auto valor:costo){
        std::cout<< valor <<std::endl;
    }

    extraccion.VectorToFile(costo,"Costo.txt");
    extraccion.EigenToFile(thetaOut,"ThetaOut.txt");
    return EXIT_SUCCESS;
}

#include "exeigennorm.h"

#include <vector>
#include <stdlib.h>
#include <cmath>
#include <boost/algorithm/string.hpp>

/*Primera funcion: Lectura de ficheros csv
 * La idea es leer linea por linea y almacenar un vector de vectores tipo String */
std::vector<std::vector<std::string>> ExEigenNorm::LeerCSV(){
    /* Se abre el archivo para lectura solamente*/
    std::fstream Archivo(setDatos);
    /*Vector de vectores del tipo String que tendra los datos del dataset*/
    std::vector<std::vector<std::string>> datosString;
    /* Se itera a traves de cada linea del dataset, y se divide el contenido
     * dado por el delimitador provisto por el constructor*/
    std::string linea ="";

    while(getline(Archivo, linea)){
        std::vector<std::string> vectorFila;
        boost::algorithm::split(vectorFila, linea, boost::is_any_of(delimitador));
        datosString.push_back(vectorFila);
    }

    /*Se cierra el archivo */
    Archivo.close();
    /*Se retorna el vector de vectores de tipo String*/
    return datosString;
}
/* Se crea la segunda funcion para guardar el vector de vectores del tipo string
    a una matriz de Eigen. Similar a Pandas para representar un data frame  */
Eigen::MatrixXd ExEigenNorm::CSVtoEigen(std::vector<std::vector<std::string>> setDatos, int filas, int col){
    /* Si tiene cabecera, se remueve
    if(header==true){
        filas-=1;
    }
    /*
     * Se itera sobre filas y columnas para almacenar en la matriz vacia (Tamaño filas*columnas), que basicamente
     * almacenara string en un vector: Luego lo pasaremos a float para ser manipulados
     */
     Eigen::MatrixXd dfMatriz(col,filas);
     for(int i = 0; i < filas; i++){
         for(int j = 0; j < col; j++){
             dfMatriz(j,i) = atof(setDatos[i][j].c_str());
         }
     }
     /*
      * Se transpone la matriz para tener filas por columnas
      */
     return dfMatriz.transpose();
}
/* A continuacion se van a implementar las funciones para la normalizacion.*/

/* En C++, la palabra clave auto especifica que el tipo de la variable que se empieza a declarar
 * se deducira automaticamente de su inicializador y, para las funciones si su tipo de retorno
 * es auto, se evaluara mediante la expresion del tipo de retorno en tiempo de ejecucion*/

/*auto ExEigenNorm::Promedio(Eigen::MatrixXd datos ){
    /* Se ingresa como entrada la matriz de datos (datos) y retorna el promedio
    return datos.colwise().mean();
}*/

/* Todavia no se sabe que retorna datos.colwise().mean(): En C++ la herencia del tipo de dato
 * no es directa o no se sabe que tipo de dato debe retornar, entonces para ello se declara el
 * tipo en una expresión "decltype" con el fin de tener seguridad de que tipo de dato
 * retornara la funcion*/

auto ExEigenNorm::Promedio(Eigen::MatrixXd datos )-> decltype(datos.colwise().mean()){
    /* Se ingresa como entrada la matriz de datos (datos) y retorna el promedio */
    return datos.colwise().mean();
}

/* Para implementar la funcion de desviacion estandar
 * datos sera igual a x_i - promedio(x)*/

auto ExEigenNorm::Desviacion(Eigen::MatrixXd datos) -> decltype(((datos.array().square().colwise().sum())/(datos.rows())).sqrt()){
    return ((datos.array().square().colwise().sum())/(datos.rows())).sqrt();
}

/*Normalizacion Z-Score es una estrategia de normalizacion de datos que evita el problema de los outliers*/

Eigen::MatrixXd ExEigenNorm::Normalizacion(Eigen::MatrixXd datos){

    Eigen::MatrixXd diferenciaPromedio = datos.rowwise()-Promedio(datos);
    //std::cout<<"Promedio: "<<std::endl<<std::endl;
    //std::cout<<Promedio(datos)<<std::endl<<std::endl;
    Eigen::MatrixXd matrizNormalizada = (diferenciaPromedio.array().rowwise())/(Desviacion(diferenciaPromedio));
    return matrizNormalizada;
}

/* A continuacion se hara una funcion para dividir los datos en conjunto de datos
 * de entrenamiento y conjunto de datos de prueba*/

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> ExEigenNorm::TrainTestSplit(Eigen::MatrixXd datos, float sizeTrain){
    int filas = datos.rows();
    int filasTrain = round(sizeTrain*filas);
    int filasTest = filas - filasTrain;
    /* Con Eigen se puede especificar un bloque de una matriz, por ejemplo
     * se pueden seleccionar las filas superiores para el conjunto de
     * entrenamiento indicando cuantas filas se desean, se selecciona desde 0
     * (fila 0) hasta el numero de filas indicado*/
    Eigen::MatrixXd entrenamiento = datos.topRows(filasTrain);

    /* Seleccionadas las filas superiores para entrenamiento, se
     * seleccionan las 11 primeras columnas (columna izquierda)
     * que representan las variables independientes FEATURES*/

    Eigen::MatrixXd X_train = entrenamiento.leftCols(datos.cols()-1);

    /*Se selecciona ahora la variable independiente que corresponde a
     * la ultima columna
     */

    Eigen::MatrixXd y_train = entrenamiento.rightCols(1);

    /* Se realiza lo mismo para el conjunto de pruebas */
    Eigen::MatrixXd test = datos.bottomRows(filasTest);

    Eigen::MatrixXd X_test = test.leftCols(datos.cols()-1);

    Eigen::MatrixXd y_test = test.rightCols(1);

    /*Finalmente se retorna una tupla dada por el conjunto de datos
     * de prueba y de entrenamiento*/

    return std::make_tuple(X_train, y_train, X_test, y_test);
}

    /*
     * Se implementan 2 funciones para exportar a ficheros desde vector y desde eigen
     */
    void ExEigenNorm::VectorToFile(std::vector<float> vector, std::string nombre){
        std::ofstream fichero(nombre);
        std::ostream_iterator<float> iterador(fichero,"\n");
        std::copy(vector.begin(),vector.end(),iterador);
    }

    void ExEigenNorm::EigenToFile(Eigen::MatrixXd datos, std::string nombre){
        std::ofstream fichero(nombre);
        if(fichero.is_open()){
            fichero<<datos<<"\n";
        }
    }




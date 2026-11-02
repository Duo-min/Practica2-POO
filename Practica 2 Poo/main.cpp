#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <iomanip>

using namespace std;

// Clase de Regresión Lineal Múltiple programada desde cero
class LinearRegression {
private:
    vector<double> weights;
    double bias;
    vector<double> mean_X;
    vector<double> std_X;
    bool is_scaled;

public:
    LinearRegression() : bias(0.0), is_scaled(false) {}

    // Métodos Getters
    vector<double> getWeights() const { return weights; }
    double getBias() const { return bias; }

    // Escalado de datos (Z-score normalization)
    vector<vector<double>> data_scaling(const vector<vector<double>>& X, bool fit_scaler = true) {
        int n_samples = X.size();
        int n_features = X[0].size();
        
        if (fit_scaler) {
            mean_X.assign(n_features, 0.0);
            std_X.assign(n_features, 0.0);

            for (int j = 0; j < n_features; ++j) {
                double sum = 0.0;
                for (int i = 0; i < n_samples; ++i) sum += X[i][j];
                mean_X[j] = sum / n_samples;
            }

            for (int j = 0; j < n_features; ++j) {
                double sum_sq = 0.0;
                for (int i = 0; i < n_samples; ++i) {
                    sum_sq += pow(X[i][j] - mean_X[j], 2);
                }
                std_X[j] = sqrt(sum_sq / n_samples);
                if (std_X[j] == 0) std_X[j] = 1.0;
            }
            is_scaled = true;
        }

        vector<vector<double>> X_scaled = X;
        for (int i = 0; i < n_samples; ++i) {
            for (int j = 0; j < n_features; ++j) {
                X_scaled[i][j] = (X[i][j] - mean_X[j]) / std_X[j];
            }
        }
        return X_scaled;
    }

    // Entrenamiento del modelo (fit) mediante Gradiente Descendente
    void fit(const vector<vector<double>>& X, const vector<double>& y, double lr = 0.05, int epochs = 3000) {
        int n_samples = X.size();
        int n_features = X[0].size();

        weights.assign(n_features, 0.0);
        bias = 0.0;

        for (int epoch = 0; epoch < epochs; ++epoch) {
            vector<double> dw(n_features, 0.0);
            double db = 0.0;

            for (int i = 0; i < n_samples; ++i) {
                double y_pred = bias;
                for (int j = 0; j < n_features; ++j) {
                    y_pred += weights[j] * X[i][j];
                }
                double error = y_pred - y[i];
                for (int j = 0; j < n_features; ++j) {
                    dw[j] += error * X[i][j];
                }
                db += error;
            }

            for (int j = 0; j < n_features; ++j) {
                weights[j] -= (lr / n_samples) * dw[j];
            }
            bias -= (lr / n_samples) * db;
        }
    }

    // Predicción
    vector<double> predict(const vector<vector<double>>& X) const {
        int n_samples = X.size();
        int n_features = X[0].size();
        vector<double> predictions(n_samples, 0.0);

        for (int i = 0; i < n_samples; ++i) {
            double y_pred = bias;
            for (int j = 0; j < n_features; ++j) {
                y_pred += weights[j] * X[i][j];
            }
            predictions[i] = y_pred;
        }
        return predictions;
    }

    // Cálculo del R2 Score
    double score(const vector<double>& y_true, const vector<double>& y_pred) const {
        int n = y_true.size();
        double ss_tot = 0.0, ss_res = 0.0, sum_y = 0.0;

        for (double val : y_true) sum_y += val;
        double mean_y = sum_y / n;

        for (int i = 0; i < n; ++i) {
            ss_res += pow(y_true[i] - y_pred[i], 2);
            ss_tot += pow(y_true[i] - mean_y, 2);
        }
        return 1.0 - (ss_res / ss_tot);
    }

    // Error Cuadrático Medio (MSE)
    double compute_mse(const vector<double>& y_true, const vector<double>& y_pred) const {
        int n = y_true.size();
        double mse = 0.0;
        for (int i = 0; i < n; ++i) {
            mse += pow(y_true[i] - y_pred[i], 2);
        }
        return mse / n;
    }
};

// Función auxiliar para lectura de archivos CSV
bool loadCSV(const string& filename, vector<vector<double>>& X, vector<double>& y, bool has_header = true) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error al abrir el archivo: " << filename << endl;
        return false;
    }

    string line;
    if (has_header && getline(file, line)) {} // Ignorar encabezado

    while (getline(file, line)) {
        stringstream ss(line);
        string cell;
        vector<double> row;

        while (getline(ss, cell, ',')) {
            if (!cell.empty()) {
                row.push_back(stod(cell));
            }
        }

        if (!row.empty()) {
            y.push_back(row.back());
            row.pop_back();
            X.push_back(row);
        }
    }
    file.close();
    return true;
}

// Objeto en Memoria Global (Requisito punto 3.2)
LinearRegression global_model;

void run_memory_analysis() {
    cout << "\n==========================================" << endl;
    cout << "SECCION 3.2: ANALISIS DE MEMORIA" << endl;
    cout << "==========================================" << endl;

    // Objeto en Memoria Local (Stack)
    LinearRegression local_model;
    
    // Objeto en Memoria Dinámica (Heap)
    LinearRegression* dynamic_model = new LinearRegression();

    cout << "Direcciones de Memoria:" << endl;
    cout << "  - Objeto Global:  " << &global_model << " (Segmento Data/BSS)" << endl;
    cout << "  - Objeto Local:   " << &local_model << " (Stack/Pila)" << endl;
    cout << "  - Objeto Dinamico: " << dynamic_model << " (Heap/Monticulo)" << endl;

    cout << "\nTamaños de Objetos (sizeof):" << endl;
    cout << "  - sizeof(global_model):  " << sizeof(global_model) << " bytes" << endl;
    cout << "  - sizeof(local_model):   " << sizeof(local_model) << " bytes" << endl;
    cout << "  - sizeof(dynamic_model): " << sizeof(*dynamic_model) << " bytes" << endl;

    // Vector de objetos en memoria dinámica
    int n_elements = 5;
    LinearRegression* dynamic_vector = new LinearRegression[n_elements];
    cout << "\nAnalisis de Vector Dinamico:" << endl;
    cout << "  - Direccion del vector: " << dynamic_vector << endl;
    cout << "  - Tamaño del vector:    " << n_elements * sizeof(LinearRegression) << " bytes (5 objetos * " << sizeof(LinearRegression) << " bytes)" << endl;

    // Liberación de memoria
    delete dynamic_model;
    delete[] dynamic_vector;
}

int main() {
    cout << "==========================================" << endl;
    cout << "PRACTICA 2: PROGRAMACION ORIENTADA A OBJETOS" << endl;
    cout << "Framework de Machine Learning en C++" << endl;
    cout << "==========================================" << endl;

    // DATASET 1: Ice Cream Selling Data
    cout << "\n>>> DATASET 1: Ice Cream Selling Data <<<" << endl;
    vector<vector<double>> X_ice;
    vector<double> y_ice;

    if (loadCSV("Ice_cream_selling_data.csv", X_ice, y_ice, true)) {
        LinearRegression mlr_ice;
        vector<vector<double>> X_ice_scaled = mlr_ice.data_scaling(X_ice, true);
        mlr_ice.fit(X_ice_scaled, y_ice, 0.05, 2000);

        vector<double> y_hat = mlr_ice.predict(X_ice_scaled);
        cout << "Pesos (Weights): " << mlr_ice.getWeights()[0] << endl;
        cout << "Sesgo (Bias): " << mlr_ice.getBias() << endl;
        cout << "Error MSE: " << mlr_ice.compute_mse(y_ice, y_hat) << endl;
        cout << "R2 Score: " << mlr_ice.score(y_ice, y_hat) << endl;
    }

    // DATASET 2: Student Exam Scores
    cout << "\n>>> DATASET 2: Student Exam Scores <<<" << endl;
    vector<vector<double>> X_student;
    vector<double> y_student;

    if (loadCSV("student_exam_scores.csv", X_student, y_student, true)) {
        LinearRegression mlr_student;
        vector<vector<double>> X_student_scaled = mlr_student.data_scaling(X_student, true);
        mlr_student.fit(X_student_scaled, y_student, 0.05, 3000);

        vector<double> y_hat = mlr_student.predict(X_student_scaled);
        cout << "Pesos (Weights): ";
        for (double w : mlr_student.getWeights()) cout << w << " ";
        cout << "\nSesgo (Bias): " << mlr_student.getBias() << endl;
        cout << "Error MSE: " << mlr_student.compute_mse(y_student, y_hat) << endl;
        cout << "R2 Score: " << mlr_student.score(y_student, y_hat) << endl;
    }

    // Ejecución del análisis de memoria
    run_memory_analysis();

    return 0;
}
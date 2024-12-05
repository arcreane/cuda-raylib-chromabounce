#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <raylib.h>
#include <cmath>
#include <iostream>
#include <ctime>
#include <vector>
#include <algorithm>
#include <string>

// Macro pour vérifier les erreurs CUDA
#define cudaCheckError(call)                                                        \
    {                                                                               \
        cudaError_t err = call;                                                     \
        if (err != cudaSuccess) {                                                   \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;    \
            exit(EXIT_FAILURE);                                                     \
        }                                                                           \
    }

// Constantes globales
const int screenWidth = 1200;
const int screenHeight = 800;
const int gameDuration = 45;  // Durée du jeu en secondes

// Structure pour stocker une particule
struct Particle {
    float x, y;
    float vx, vy;
    int r, g, b;
    bool active;
    bool interacted; // Ajout pour savoir si la particule a été affectée par la souris
};

// Structure pour stocker un score
struct Score {
    std::string playerName;
    int score;
};

// Variables globales pour la simulation des particules
int numParticles = 1000;
float particleSize = 2.0f;
float particleSpeed = 100.0f;
int zoneSize = 100;

// Fonction pour comparer les scores
bool compareScores(const Score& a, const Score& b) {
    return a.score > b.score;
}

// Kernel CUDA pour initialiser les particules
__global__ void initParticlesKernel(Particle* particles, int numParticles, int screenWidth, int screenHeight, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles) {
        curandState state;
        curand_init(seed, idx, 0, &state);

        particles[idx].x = curand_uniform(&state) * screenWidth;
        particles[idx].y = curand_uniform(&state) * screenHeight;
        particles[idx].vx = curand_uniform(&state) * 2.0f - 1.0f;
        particles[idx].vy = curand_uniform(&state) * 2.0f - 1.0f;
        particles[idx].r = (int)(curand_uniform(&state) * 256);
        particles[idx].g = (int)(curand_uniform(&state) * 256);
        particles[idx].b = (int)(curand_uniform(&state) * 256);
        particles[idx].active = true;
        particles[idx].interacted = false;
    }
}

// Kernel CUDA pour mettre à jour les particules avec gestion des collisions
__global__ void updateParticlesKernel(
    Particle* particles, int numParticles, float particleSize, float particleSpeed, float timeStep,
    int screenWidth, int screenHeight, int* score, int zoneX, int zoneY, int zoneSize,
    float mouseX, float mouseY, bool isAttracting, bool isRepelling) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles || !particles[idx].active) return;

    // Interactions avec la souris
    float fx = 0.0f, fy = 0.0f;
    float dx = mouseX - particles[idx].x;
    float dy = mouseY - particles[idx].y;
    float distance = sqrtf(dx * dx + dy * dy);

    if (distance < 100.0f && distance > 0.0f) {
        float force = 10.0f / distance;
        particles[idx].interacted = true; // Marque comme affectée par la souris
        if (isAttracting) {
            fx += force * dx;
            fy += force * dy;
        }
        else if (isRepelling) {
            fx -= force * dx;
            fy -= force * dy;
        }
    }

    particles[idx].vx += fx * timeStep;
    particles[idx].vy += fy * timeStep;

    // Mise à jour de la position
    particles[idx].x += particles[idx].vx * particleSpeed * timeStep;
    particles[idx].y += particles[idx].vy * particleSpeed * timeStep;

    // Gestion des collisions avec les bords
    if (particles[idx].x < 0 || particles[idx].x > screenWidth) particles[idx].vx *= -1.0f;
    if (particles[idx].y < 0 || particles[idx].y > screenHeight) particles[idx].vy *= -1.0f;

    particles[idx].x = fminf(fmaxf(particles[idx].x, 0.0f), screenWidth);
    particles[idx].y = fminf(fmaxf(particles[idx].y, 0.0f), screenHeight);

    // Gestion des collisions entre particules
    for (int i = 0; i < numParticles; i++) {
        if (i == idx || !particles[i].active) continue;

        float distX = particles[i].x - particles[idx].x;
        float distY = particles[i].y - particles[idx].y;
        float dist = sqrtf(distX * distX + distY * distY);

        if (dist < particleSize * 2.0f && dist > 0.0f) {
            // Calcul de la direction opposée
            float normX = distX / dist;
            float normY = distY / dist;

            // Inversion des vitesses
            particles[idx].vx -= normX * particleSpeed * 0.1f;
            particles[idx].vy -= normY * particleSpeed * 0.1f;
            particles[i].vx += normX * particleSpeed * 0.1f;
            particles[i].vy += normY * particleSpeed * 0.1f;

            // Séparation
            particles[idx].x -= normX * 0.5f;
            particles[idx].y -= normY * 0.5f;
            particles[i].x += normX * 0.5f;
            particles[i].y += normY * 0.5f;
        }
    }

    // Vérifie si la particule est dans la zone d'embut et a été affectée
    if (particles[idx].x > zoneX && particles[idx].x < zoneX + zoneSize &&
        particles[idx].y > zoneY && particles[idx].y < zoneY + zoneSize &&
        particles[idx].interacted) {
        atomicAdd(score, 1);
        particles[idx].active = false;
    }
}

// Fonction pour gérer le menu principal
int showMenu() {
    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_ESCAPE)) return -1;

        BeginDrawing();
        ClearBackground(BLACK);

        DrawText("Choisissez une difficulté:", screenWidth / 2 - 150, screenHeight / 2 - 100, 20, WHITE);
        DrawRectangle(screenWidth / 2 - 150, screenHeight / 2, 100, 50, LIGHTGRAY);
        DrawText("Facile", screenWidth / 2 - 140, screenHeight / 2 + 10, 20, BLACK);
        DrawRectangle(screenWidth / 2 - 50, screenHeight / 2, 100, 50, GRAY);
        DrawText("Moyen", screenWidth / 2 - 40, screenHeight / 2 + 10, 20, BLACK);
        DrawRectangle(screenWidth / 2 + 50, screenHeight / 2, 100, 50, DARKGRAY);
        DrawText("Difficile", screenWidth / 2 + 60, screenHeight / 2 + 10, 20, WHITE);

        EndDrawing();

        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
            Vector2 mousePos = GetMousePosition();
            if (CheckCollisionPointRec(mousePos, { float(screenWidth / 2 - 150), float(screenHeight / 2), 100, 50 })) {
                numParticles = 500;
                particleSize = 10.0f;
                particleSpeed = 50.0f;
                zoneSize = 150;
                return 1;
            }
            else if (CheckCollisionPointRec(mousePos, { float(screenWidth / 2 - 50), float(screenHeight / 2), 100, 50 })) {
                numParticles = 1000;
                particleSize = 4.0f;
                particleSpeed = 100.0f;
                zoneSize = 100;
                return 2;
            }
            else if (CheckCollisionPointRec(mousePos, { float(screenWidth / 2 + 50), float(screenHeight / 2), 100, 50 })) {
                numParticles = 1500;
                particleSize = 1.0f;
                particleSpeed = 200.0f;
                zoneSize = 50;
                return 3;
            }
        }
    }
    return -1;
}

// Fonction principale
int main() {
    InitWindow(screenWidth, screenHeight, "Simulation de Particules Gamifié");
    SetTargetFPS(60);

    std::vector<Score> highScores;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_ESCAPE)) break;

        int difficulty = showMenu();
        if (difficulty == -1) break;

        Particle* h_particles = new Particle[numParticles];
        Particle* d_particles;
        cudaCheckError(cudaMalloc(&d_particles, numParticles * sizeof(Particle)));

        int* d_score;
        cudaCheckError(cudaMalloc(&d_score, sizeof(int)));
        cudaCheckError(cudaMemset(d_score, 0, sizeof(int)));

        int threadsPerBlock = 256;
        int blocksPerGrid = (numParticles + threadsPerBlock - 1) / threadsPerBlock;

        unsigned long long seed = static_cast<unsigned long long>(time(nullptr));
        initParticlesKernel << <blocksPerGrid, threadsPerBlock >> > (d_particles, numParticles, screenWidth, screenHeight, seed);
        cudaCheckError(cudaDeviceSynchronize());

        int score = 0;
        time_t startTime = time(nullptr);

        while (!WindowShouldClose()) {
            if (IsKeyPressed(KEY_ESCAPE)) break;

            int elapsedTime = static_cast<int>(time(nullptr) - startTime);
            if (elapsedTime >= gameDuration) break;

            Vector2 mousePosition = GetMousePosition();
            bool isAttracting = IsMouseButtonDown(MOUSE_LEFT_BUTTON);
            bool isRepelling = IsMouseButtonDown(MOUSE_RIGHT_BUTTON);

            updateParticlesKernel << <blocksPerGrid, threadsPerBlock >> > (
                d_particles, numParticles, particleSize, particleSpeed, 0.02f,
                screenWidth, screenHeight, d_score, screenWidth / 2 - zoneSize / 2,
                screenHeight / 2 - zoneSize / 2, zoneSize, mousePosition.x, mousePosition.y,
                isAttracting, isRepelling);
            cudaCheckError(cudaDeviceSynchronize());

            cudaMemcpy(h_particles, d_particles, numParticles * sizeof(Particle), cudaMemcpyDeviceToHost);
            cudaMemcpy(&score, d_score, sizeof(int), cudaMemcpyDeviceToHost);

            BeginDrawing();
            ClearBackground(BLACK);

            DrawRectangle(screenWidth / 2 - zoneSize / 2, screenHeight / 2 - zoneSize / 2, zoneSize, zoneSize, BLUE);

            for (int i = 0; i < numParticles; ++i) {
                if (h_particles[i].active) {
                    DrawCircle(h_particles[i].x, h_particles[i].y, particleSize, { (unsigned char)h_particles[i].r, (unsigned char)h_particles[i].g, (unsigned char)h_particles[i].b, 255 });
                }
            }

            DrawText(TextFormat("Score: %d", score), 10, 10, 20, WHITE);
            DrawText(TextFormat("Time: %d/%d seconds", elapsedTime, gameDuration), 10, 40, 20, WHITE);

            EndDrawing();
        }

        delete[] h_particles;
        cudaFree(d_particles);
        cudaFree(d_score);

        highScores.push_back({ "Player", score });
        std::sort(highScores.begin(), highScores.end(), compareScores);
        if (highScores.size() > 6) highScores.resize(6);
    }

    CloseWindow();
    return 0;
}

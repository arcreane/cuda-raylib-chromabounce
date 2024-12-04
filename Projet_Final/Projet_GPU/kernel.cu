#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <raylib.h>
#include <cmath>
#include <iostream>
#include <ctime>

// Macro pour vérifier les erreurs CUDA
#define cudaCheckError(call)                                                        \
    {                                                                               \
        cudaError_t err = call;                                                     \
        if (err != cudaSuccess) {                                                   \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;    \
            exit(EXIT_FAILURE);                                                     \
        }                                                                           \
    }

// Constantes globales pour CUDA
__constant__ float maxVelocity = 100.0f;       // Limite de vitesse
__constant__ float forceStrength = 10.0f;     // Intensité de la force

// Constantes globales pour le CPU
const int numParticles = 1000;
const float timeStep = 0.02f;
const float interactionRadius = 100.0f;
const int screenWidth = 1200;
const int screenHeight = 800;
const int gameDuration = 45;  // Durée du jeu en secondes

// Zone de marquage
const int zoneSize = 100;       // Taille du carré bleu
const int zoneX = screenWidth / 2 - zoneSize / 2;
const int zoneY = screenHeight / 2 - zoneSize / 2;

// Structure pour représenter une particule
struct Particle {
    float x, y;     // Position 2D
    float vx, vy;   // Vitesse
    int r, g, b;    // Couleurs RGB
    bool active;    // Indique si la particule est active
};

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
    }
}

// Kernel CUDA pour mettre à jour les particules
__global__ void updateParticlesKernel(Particle* particles, int numParticles, float interactionRadius, float timeStep, int screenWidth, int screenHeight, float mouseX, float mouseY, bool isAttracting, bool isRepelling, int* score, int zoneX, int zoneY, int zoneSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    if (!particles[idx].active) return; // Particule déjà inactive

    // Vérifie si la particule est dans la zone de marquage
    if (particles[idx].x > zoneX && particles[idx].x < zoneX + zoneSize &&
        particles[idx].y > zoneY && particles[idx].y < zoneY + zoneSize) {
        atomicAdd(score, 1);           // Augmente le score
        particles[idx].active = false; // Désactive la particule
        return;
    }

    float fx = 0.0f, fy = 0.0f;

    // Interaction avec le champ contrôlé par la souris
    float dx = mouseX - particles[idx].x;
    float dy = mouseY - particles[idx].y;
    float distance = sqrtf(dx * dx + dy * dy);

    if (distance < interactionRadius && distance > 0.0f) {
        float force = forceStrength / distance;
        if (isAttracting) {
            fx += force * dx;
            fy += force * dy;
        }
        else if (isRepelling) {
            fx -= force * dx;
            fy -= force * dy;
        }
    }

    // Mise à jour de la vitesse et limitation
    particles[idx].vx += fx * timeStep;
    particles[idx].vy += fy * timeStep;

    float velocity = sqrtf(particles[idx].vx * particles[idx].vx + particles[idx].vy * particles[idx].vy);
    if (velocity > maxVelocity) {
        particles[idx].vx *= maxVelocity / velocity;
        particles[idx].vy *= maxVelocity / velocity;
    }

    // Mise à jour de la position
    particles[idx].x += particles[idx].vx * timeStep;
    particles[idx].y += particles[idx].vy * timeStep;

    // Gestion des collisions avec les bords
    if (particles[idx].x < 0 || particles[idx].x > screenWidth) particles[idx].vx *= -1.0f;
    if (particles[idx].y < 0 || particles[idx].y > screenHeight) particles[idx].vy *= -1.0f;

    particles[idx].x = fminf(fmaxf(particles[idx].x, 0.0f), screenWidth);
    particles[idx].y = fminf(fmaxf(particles[idx].y, 0.0f), screenHeight);
}

int main() {
    // Initialisation de Raylib
    InitWindow(screenWidth, screenHeight, "Simulation de Particules Gamifiée avec Zone de Marquage");
    SetTargetFPS(60);

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
        // Chronomètre
        int elapsedTime = static_cast<int>(time(nullptr) - startTime);
        if (elapsedTime >= gameDuration) {
            break; // Fin du jeu après 45 secondes
        }

        Vector2 mousePosition = GetMousePosition();
        bool isAttracting = IsMouseButtonDown(MOUSE_LEFT_BUTTON);
        bool isRepelling = IsMouseButtonDown(MOUSE_RIGHT_BUTTON);

        updateParticlesKernel << <blocksPerGrid, threadsPerBlock >> > (d_particles, numParticles, interactionRadius, timeStep, screenWidth, screenHeight, mousePosition.x, mousePosition.y, isAttracting, isRepelling, d_score, zoneX, zoneY, zoneSize);
        cudaCheckError(cudaDeviceSynchronize());

        cudaCheckError(cudaMemcpy(h_particles, d_particles, numParticles * sizeof(Particle), cudaMemcpyDeviceToHost));
        cudaCheckError(cudaMemcpy(&score, d_score, sizeof(int), cudaMemcpyDeviceToHost));

        BeginDrawing();
        ClearBackground(BLACK);

        // Dessiner la zone de marquage
        DrawRectangle(zoneX, zoneY, zoneSize, zoneSize, BLUE);

        // Dessiner les particules actives
        for (int i = 0; i < numParticles; ++i) {
            if (h_particles[i].active) {
                Color particleColor = { (unsigned char)h_particles[i].r, (unsigned char)h_particles[i].g, (unsigned char)h_particles[i].b, 255 };
                DrawCircle(h_particles[i].x, h_particles[i].y, 2, particleColor);
            }
        }

        DrawText(TextFormat("Score: %d", score), 10, 10, 20, WHITE);
        DrawText(TextFormat("Time: %d/%d seconds", elapsedTime, gameDuration), 10, 40, 20, WHITE);
        EndDrawing();
    }

    // Afficher le score final
    while (!WindowShouldClose()) {
        BeginDrawing();
        ClearBackground(BLACK);
        DrawText("FIN DU JEU", screenWidth / 2 - 100, screenHeight / 2 - 50, 40, WHITE);
        DrawText(TextFormat("Votre score: %d", score), screenWidth / 2 - 100, screenHeight / 2, 30, WHITE);
        DrawText("Appuyez sur ESC pour quitter.", screenWidth / 2 - 150, screenHeight / 2 + 50, 20, WHITE);
        EndDrawing();
    }

    delete[] h_particles;
    cudaCheckError(cudaFree(d_particles));
    cudaCheckError(cudaFree(d_score));
    CloseWindow();

    return 0;
}

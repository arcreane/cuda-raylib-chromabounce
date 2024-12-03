#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <curand_kernel.h>  // Pour les nombres aléatoires sur GPU
#include <raylib.h>
#include <cmath>  // Pour sqrtf
#include <iostream>

// Structure pour représenter une particule
struct Particle {
    float x, y;     // Position
    float vx, vy;   // Vitesse
    int r, g, b;    // Couleurs RGB
};

// Constantes globales
const int numParticles = 1000;
const float timeStep = 0.01f;
const float interactionRadius = 10.0f;
const float forceStrength = 0.1f;
const int screenWidth = 1200;  // Taille augmentée
const int screenHeight = 800; // Taille augmentée

// Kernel CUDA pour initialiser les particules
__global__ void initParticlesKernel(Particle* particles, int numParticles, int screenWidth, int screenHeight, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles) {
        // Initialisation de curand
        curandState state;
        curand_init(seed, idx, 0, &state);

        // Génération des valeurs aléatoires
        particles[idx].x = curand_uniform(&state) * screenWidth;
        particles[idx].y = curand_uniform(&state) * screenHeight;
        particles[idx].vx = 0.0f;  // Initialement 0
        particles[idx].vy = 0.0f;
        particles[idx].r = (int)(curand_uniform(&state) * 256); // Rouge
        particles[idx].g = (int)(curand_uniform(&state) * 256); // Vert
        particles[idx].b = (int)(curand_uniform(&state) * 256); // Bleu
    }
}

// Kernel CUDA pour mettre à jour les particules
__global__ void updateParticlesKernel(Particle* particles, int numParticles, float interactionRadius, float forceStrength, float timeStep, int screenWidth, int screenHeight) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    float fx = 0.0f, fy = 0.0f;

    for (int j = 0; j < numParticles; ++j) {
        if (j == idx) continue;

        float dx = particles[j].x - particles[idx].x;
        float dy = particles[j].y - particles[idx].y;
        float distance = sqrtf(dx * dx + dy * dy);

        if (distance < interactionRadius && distance > 0.0f) {
            float force = forceStrength / distance;
            fx += force * dx;
            fy += force * dy;
        }
    }

    particles[idx].vx += fx * timeStep;
    particles[idx].vy += fy * timeStep;

    particles[idx].x += particles[idx].vx * timeStep;
    particles[idx].y += particles[idx].vy * timeStep;

    // Gestion des collisions avec les bords
    if (particles[idx].x < 0 || particles[idx].x > screenWidth) particles[idx].vx *= -1.0f;
    if (particles[idx].y < 0 || particles[idx].y > screenHeight) particles[idx].vy *= -1.0f;

    // Garde les particules dans les limites
    particles[idx].x = fminf(fmaxf(particles[idx].x, 0.0f), screenWidth);
    particles[idx].y = fminf(fmaxf(particles[idx].y, 0.0f), screenHeight);
}

int main() {
    // Initialisation de Raylib
    InitWindow(screenWidth, screenHeight, "Simulation de Particules");
    SetTargetFPS(60);

    // Initialisation des particules
    Particle* h_particles = new Particle[numParticles];
    Particle* d_particles;
    cudaMalloc(&d_particles, numParticles * sizeof(Particle));

    int threadsPerBlock = 256;
    int blocksPerGrid = (numParticles + threadsPerBlock - 1) / threadsPerBlock;

    // Seed pour la génération aléatoire
    unsigned long long seed = static_cast<unsigned long long>(time(nullptr));

    // Initialiser les particules sur le GPU
    initParticlesKernel << <blocksPerGrid, threadsPerBlock >> > (d_particles, numParticles, screenWidth, screenHeight, seed);
    cudaDeviceSynchronize();

    while (!WindowShouldClose()) {
        // Mise à jour des particules avec CUDA
        updateParticlesKernel << <blocksPerGrid, threadsPerBlock >> > (d_particles, numParticles, interactionRadius, forceStrength, timeStep, screenWidth, screenHeight);
        cudaDeviceSynchronize();

        // Copier les données mises à jour du GPU au CPU
        cudaMemcpy(h_particles, d_particles, numParticles * sizeof(Particle), cudaMemcpyDeviceToHost);

        // Rendu graphique
        BeginDrawing();
        ClearBackground(BLACK);

        for (int i = 0; i < numParticles; ++i) {
            Color particleColor = { (unsigned char)h_particles[i].r, (unsigned char)h_particles[i].g, (unsigned char)h_particles[i].b, 255 };
            DrawCircle(h_particles[i].x, h_particles[i].y, 2, particleColor);
        }

        DrawText("Appuyez sur ESC pour quitter.", 10, 10, 20, WHITE);
        EndDrawing();
    }

    // Libération de la mémoire
    delete[] h_particles;
    cudaFree(d_particles);
    CloseWindow();

    return 0;
}

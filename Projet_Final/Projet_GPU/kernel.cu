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
const int numParticles = 500;       // Réduction du nombre de particules pour la fluidité
const float timeStep = 0.2f;        // Augmentation drastique du pas de temps
const float interactionRadius = 10.0f; // Rayon réduit pour des calculs plus rapides
const float forceStrength = 0.5f;   // Force augmentée pour un mouvement plus dynamique
const int screenWidth = 800;
const int screenHeight = 600;

// Kernel CUDA pour initialiser les particules
__global__ void initParticlesKernel(Particle* particles, int numParticles, int screenWidth, int screenHeight, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles) {
        curandState state;
        curand_init(seed, idx, 0, &state);

        particles[idx].x = curand_uniform(&state) * screenWidth;
        particles[idx].y = curand_uniform(&state) * screenHeight;
        particles[idx].vx = curand_uniform(&state) * 2.0f - 1.0f; // Vitesse initiale aléatoire (-1 à 1)
        particles[idx].vy = curand_uniform(&state) * 2.0f - 1.0f;
        particles[idx].r = (int)(curand_uniform(&state) * 256);
        particles[idx].g = (int)(curand_uniform(&state) * 256);
        particles[idx].b = (int)(curand_uniform(&state) * 256);
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

    if (particles[idx].x < 0 || particles[idx].x > screenWidth) particles[idx].vx *= -1.0f;
    if (particles[idx].y < 0 || particles[idx].y > screenHeight) particles[idx].vy *= -1.0f;

    particles[idx].x = fminf(fmaxf(particles[idx].x, 0.0f), screenWidth);
    particles[idx].y = fminf(fmaxf(particles[idx].y, 0.0f), screenHeight);
}

int main() {
    // Initialisation de Raylib
    InitWindow(screenWidth, screenHeight, "Simulation de Particules - Très Rapide");
    SetTargetFPS(60);

    Particle* h_particles = new Particle[numParticles];
    Particle* d_particles;
    cudaMalloc(&d_particles, numParticles * sizeof(Particle));

    int threadsPerBlock = 256;
    int blocksPerGrid = (numParticles + threadsPerBlock - 1) / threadsPerBlock;

    unsigned long long seed = static_cast<unsigned long long>(time(nullptr));

    initParticlesKernel << <blocksPerGrid, threadsPerBlock >> > (d_particles, numParticles, screenWidth, screenHeight, seed);
    cudaDeviceSynchronize();

    while (!WindowShouldClose()) {
        updateParticlesKernel << <blocksPerGrid, threadsPerBlock >> > (d_particles, numParticles, interactionRadius, forceStrength, timeStep, screenWidth, screenHeight);
        cudaDeviceSynchronize();

        cudaMemcpy(h_particles, d_particles, numParticles * sizeof(Particle), cudaMemcpyDeviceToHost);

        BeginDrawing();
        ClearBackground(BLACK);

        for (int i = 0; i < numParticles; ++i) {
            Color particleColor = { (unsigned char)h_particles[i].r, (unsigned char)h_particles[i].g, (unsigned char)h_particles[i].b, 255 };
            DrawCircle(h_particles[i].x, h_particles[i].y, 2, particleColor);
        }

        DrawText("Appuyez sur ESC pour quitter.", 10, 10, 20, WHITE);
        EndDrawing();
    }

    delete[] h_particles;
    cudaFree(d_particles);
    CloseWindow();

    return 0;
}

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <cmath>  // Pour sqrt et pow
#include <raylib.h> // Bibliothèque Raylib

// Structure pour représenter une particule
struct Particle {
    float x, y;     // Position
    float vx, vy;   // Vitesse
    int color;      // Couleur
};

// Constantes globales
const int numParticles = 1000;
const float timeStep = 0.01f;
const float interactionRadius = 10.0f;
const float forceStrength = 0.1f;
const int screenWidth = 800;
const int screenHeight = 600;

// Kernel CUDA pour initialiser les particules
__global__ void initParticlesKernel(Particle* particles, int numParticles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles) {
        particles[idx].x = idx % 100;       // Position X
        particles[idx].y = idx / 100;       // Position Y
        particles[idx].vx = 0.0f;           // Vitesse X
        particles[idx].vy = 0.0f;           // Vitesse Y
        particles[idx].color = idx % 256;   // Couleur
    }
}

// Kernel CUDA pour mettre à jour les particules
__global__ void updateParticlesKernel(Particle* particles, int numParticles) {
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

    initParticlesKernel << <blocksPerGrid, threadsPerBlock >> > (d_particles, numParticles);
    cudaDeviceSynchronize();

    while (!WindowShouldClose()) {
        // Mise à jour des particules avec CUDA
        updateParticlesKernel << <blocksPerGrid, threadsPerBlock >> > (d_particles, numParticles);
        cudaDeviceSynchronize();

        // Copier les particules sur le CPU pour affichage
        cudaMemcpy(h_particles, d_particles, numParticles * sizeof(Particle), cudaMemcpyDeviceToHost);

        // Rendu graphique
        BeginDrawing();
        ClearBackground(BLACK);

        for (int i = 0; i < numParticles; ++i) {
            Color particleColor = { h_particles[i].color, 50, 200, 255 };
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
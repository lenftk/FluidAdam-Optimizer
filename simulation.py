import pygame
import numpy as np

try:
    import cupy as cp
    test_array = cp.array([1, 2, 3])
    test_result = test_array + 1
    GPU_AVAILABLE = True
except Exception as e:
    GPU_AVAILABLE = False
    cp = np

SIZE = 100
SCALE = 3
ITER = 2

pygame.init()
screen = pygame.display.set_mode((SIZE * SCALE, SIZE * SCALE))
pygame.display.set_caption("Navier-Stokes Fluid Simulation")
clock = pygame.time.Clock()

class Fluid:
    def __init__(self, dt, diffusion, viscosity):
        self.size = SIZE
        self.dt = dt
        self.diff = diffusion
        self.visc = viscosity
        
        self.xp = cp if GPU_AVAILABLE else np
        self.vx = self.xp.zeros((self.size, self.size))
        self.vy = self.xp.zeros((self.size, self.size))
        self.vx0 = self.xp.zeros((self.size, self.size))
        self.vy0 = self.xp.zeros((self.size, self.size))
        self.density = self.xp.zeros((self.size, self.size))
        self.s = self.xp.zeros((self.size, self.size))

    def step(self):
        self.diffuse(1, self.vx0, self.vx, self.visc)
        self.diffuse(2, self.vy0, self.vy, self.visc)
        
        self.project(self.vx0, self.vy0, self.vx, self.vy)
        
        self.advect(1, self.vx, self.vx0, self.vx0, self.vy0)
        self.advect(2, self.vy, self.vy0, self.vx0, self.vy0)
        self.project(self.vx, self.vy, self.vx0, self.vy0)
        self.diffuse(0, self.s, self.density, self.diff)
        self.advect(0, self.density, self.s, self.vx, self.vy)

    def add_density(self, x, y, amount):
        self.density[x, y] += amount

    def add_velocity(self, x, y, amount_x, amount_y):
        self.vx[x, y] += amount_x
        self.vy[x, y] += amount_y

    def set_bnd(self, b, x):
        x[0, :] = -x[1, :] if b == 1 else x[1, :]
        x[self.size-1, :] = -x[self.size-2, :] if b == 1 else x[self.size-2, :]
        x[:, 0] = -x[:, 1] if b == 2 else x[:, 1]
        x[:, self.size-1] = -x[:, self.size-2] if b == 2 else x[:, self.size-2]
        
        x[0, 0] = 0.5 * (x[1, 0] + x[0, 1])
        x[0, self.size-1] = 0.5 * (x[1, self.size-1] + x[0, self.size-2])
        x[self.size-1, 0] = 0.5 * (x[self.size-2, 0] + x[self.size-1, 1])
        x[self.size-1, self.size-1] = 0.5 * (x[self.size-2, self.size-1] + x[self.size-1, self.size-2])

    def lin_solve(self, b, x, x0, a, c):
        c_recip = 1.0 / c
        for _ in range(ITER):
            x[1:-1, 1:-1] = (x0[1:-1, 1:-1] + a * (x[:-2, 1:-1] + x[2:, 1:-1] + x[1:-1, :-2] + x[1:-1, 2:])) * c_recip
            self.set_bnd(b, x)

    def diffuse(self, b, x, x0, diff):
        a = self.dt * diff * (self.size - 2) * (self.size - 2)
        self.lin_solve(b, x, x0, a, 1 + 4 * a)

    def advect(self, b, d, d0, vx, vy):
        dtx = self.dt * (self.size - 2)
        dty = self.dt * (self.size - 2)
        
        if GPU_AVAILABLE or True:  # Using vectorized version as default
            i_grid, j_grid = self.xp.meshgrid(
                self.xp.arange(1, self.size - 1), 
                self.xp.arange(1, self.size - 1), 
                indexing='ij'
            )
            
            tmp1 = dtx * vx[1:-1, 1:-1]
            tmp2 = dty * vy[1:-1, 1:-1]
            x = i_grid - tmp1
            y = j_grid - tmp2
            
            x = self.xp.clip(x, 0.5, self.size - 1.5)
            y = self.xp.clip(y, 0.5, self.size - 1.5)
            
            i0 = self.xp.floor(x).astype(int)
            i1 = i0 + 1
            j0 = self.xp.floor(y).astype(int)
            j1 = j0 + 1
            
            s1 = x - i0
            s0 = 1 - s1
            t1 = y - j0
            t0 = 1 - t1
            
            d[1:-1, 1:-1] = (s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1]) +
                             s1 * (t0 * d0[i1, j0] + t1 * d0[i1, j1]))
        else:
            for i in range(1, self.size - 1):
                for j in range(1, self.size - 1):
                    tmp1 = dtx * vx[i, j]
                    tmp2 = dty * vy[i, j]
                    x = i - tmp1
                    y = j - tmp2
                    
                    if x < 0.5: x = 0.5
                    if x > self.size - 1.5: x = self.size - 1.5
                    i0 = int(x)
                    i1 = i0 + 1
                    
                    if y < 0.5: y = 0.5
                    if y > self.size - 1.5: y = self.size - 1.5
                    j0 = int(y)
                    j1 = j0 + 1
                    
                    s1 = x - i0
                    s0 = 1 - s1
                    t1 = y - j0
                    t0 = 1 - t1
                    
                    d[i, j] = s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1]) + \
                             s1 * (t0 * d0[i1, j0] + t1 * d0[i1, j1])
        
        self.set_bnd(b, d)

    def project(self, vx, vy, p, div):
        div[1:-1, 1:-1] = -0.5 * (vx[2:, 1:-1] - vx[:-2, 1:-1] + vy[1:-1, 2:] - vy[1:-1, :-2]) / self.size
        p.fill(0)
        self.set_bnd(0, div)
        self.set_bnd(0, p)
        self.lin_solve(0, p, div, 1, 4)
        vx[1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1] - p[:-2, 1:-1]) * self.size
        vy[1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, :-2]) * self.size
        self.set_bnd(1, vx)
        self.set_bnd(2, vy)

    def get_density_for_draw(self):
        if GPU_AVAILABLE:
            return cp.asnumpy(self.density)
        else:
            return self.density

def draw_density(fluid):
    density = fluid.get_density_for_draw()
    for i in range(fluid.size):
        for j in range(fluid.size):
            d = density[i, j]
            if d > 0.05:  
                color = (min(255, int(d * 150)), 0, max(0, int(255 - d * 150)))
                pygame.draw.rect(screen, color, (i * SCALE, j * SCALE, SCALE, SCALE))

def main():
    fluid = Fluid(0.1, 0, 0.0000001)
    running = True
    pmouse_x, pmouse_y = 0, 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        mouse_x, mouse_y = pygame.mouse.get_pos()
        if pygame.mouse.get_pressed()[0]:
            grid_x = mouse_x // SCALE
            grid_y = mouse_y // SCALE
            
            if 0 <= grid_x < SIZE and 0 <= grid_y < SIZE:
                fluid.add_density(grid_x, grid_y, 100)
                amount_x = (mouse_x - pmouse_x) * 0.5
                amount_y = (mouse_y - pmouse_y) * 0.5
                fluid.add_velocity(grid_x, grid_y, amount_x, amount_y)
        
        pmouse_x, pmouse_y = mouse_x, mouse_y
        screen.fill((0, 0, 0))
        fluid.step()
        draw_density(fluid)
        
        fluid.density *= 0.99
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    main()

import tensorflow as tf
import numpy as np



def simulate_dynamics(position, velocity, eps, n_leapfrog, energy):
	
	def leapfrog(pos, vel, eps, i):
	        dE_dpos = tf.gradients(tf.reduce_sum(energy(pos)), pos)[0]
	        new_vel = vel - eps * dE_dpos
	        new_pos = pos + eps * new_vel
	        return [new_pos, new_vel, eps, tf.add(i, 1)]

    def condition(pos, vel, eps, i):
        return tf.less(i, n_leapfrog)

    dE_dpos = tf.gradients(tf.reduce_sum(energy(initial_pos)), initial_pos)[0]
    vel_half_step = initial_vel - 0.5 * eps * dE_dpos
    pos_full_step = initial_pos + eps * vel_half_step

    i = tf.constant(0)
    final_pos, new_vel, _, _ = tf.while_loop(condition, leapfrog, [pos_full_step, vel_half_step, eps, i])
    dE_dpos = tf.gradients(tf.reduce_sum(energy(final_pos)), final_pos)[0]
    final_vel = new_vel - 0.5 * eps * dE_dpos
    return final_pos, final_vel

def hmc_move(initial_pos, energy_fn, stepsize, n_steps):
    initial_vel = tf.random_normal(tf.shape(initial_pos))
    print('simulating dynamics')
    final_pos, final_vel = simulate_dynamics(
        initial_pos=initial_pos,
        initial_vel=initial_vel,
        stepsize=stepsize,
        n_steps=n_steps,
        energy_fn=energy_fn
    )
    print('metropolis')
    accept = metropolis_hastings_accept(
        energy_prev=hamiltonian(initial_pos, initial_vel, energy_fn),
        energy_next=hamiltonian(final_pos, final_vel, energy_fn)
    )
    return accept, final_pos, final_vel
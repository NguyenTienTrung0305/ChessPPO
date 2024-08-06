# Self Play
import chess
import chess.engine
import torch
import numpy as np
import os

from Agent import Agent

def train():
    agent = Agent()

    max_training_timesteps = int(1e6)
    
    max_ep_len = 1000
    update_timestep = max_ep_len * 4
    
    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0
    print_freq = max_ep_len * 8

    log_running_reward = 0
    log_running_episodes = 0
    log_freq = max_ep_len * 2  
    
    time_step = 0
    i_episode = 0
    
    save_model_freq = 1e5
    
    ########################## LOG #######################
    log_dir = "ChessPPO/PPO_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)
    
    #### create new log file for each run
    log_f_name = log_dir + '/PPO__log_' + str(run_num) + ".csv"
    
    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward\n')
    
    #####################################################
    
    
    ################### checkpointing ###################
    random_seed = 0   
    run_num_pretrained = 0  

    directory = "ChessPPO/PPO_preTrained/"
    if not os.path.exists(directory):
          os.makedirs(directory)
          
    #####################################################
    
    while time_step < max_training_timesteps:
        # reset
        board = chess.Board()
        env = chess.engine.SimpleEngine.popen_uci("..\stockfish-windows-x86-64-avx2.exe")
        
        current_ep_reward = 0
        
        game_moves = 0
        
        for t in range(1, max_ep_len+1):
            game_moves += 1
            # select action with policy
            _, chosen_move = agent.select_action(board)

            # next state
            board.push(chosen_move)    
            done = board.result() != '*' or game_moves >= max_ep_len - 1
            
            analysis = env.analyse(board=board, limit=chess.engine.Limit(depth=5))
            reward = analysis['score'].relative.score(mate_score=10000) / 100
            
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminate.append(done)
            

            time_step += 1
            current_ep_reward += reward
            

            # update PPO agent
            if time_step % update_timestep == 0:
                agent.update()
                
            # print(board.unicode())
            # print()
            
            # log in logging file
            if time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0
            
            # printing average reward
            if time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0
                
            # save model weights
            if time_step % save_model_freq == 0:
                checkpoint_path = directory + "PPO_{}_{}.pth".format(random_seed, run_num_pretrained)
                agent.save(checkpoint_path)
                run_num_pretrained += 1
                
            if done:
                break
            
        
        print_running_reward += current_ep_reward
        print_running_episodes += 1
        
        log_running_reward += current_ep_reward
        log_running_episodes += 1
        
        i_episode += 1
                
if __name__ == '__main__':
    train()
    
                        
            

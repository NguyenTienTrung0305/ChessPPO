import torch
import torch.nn as nn
import torch.optim as optim
import chess
import numpy as np
from torch.distributions import Categorical
from ActorCritic import ActorCritic, RolloutBuffer

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
device = torch.device("cuda" if use_cuda else "cpu")


class Agent:
    def __init__(self):
        self.K_epochs = 100
        self.eps_clip = 0.2
        
        # train parameter
        self.gamma = 0.99 # discount factor
        self.lr_actor = 0.0003
        self.lr_critic =  0.001
        
        self.buffer = RolloutBuffer()
        
        self.policy = ActorCritic().to(device)
        self.optimizer = optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': self.lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': self.lr_critic}
        ])
        
        self.policy_old = ActorCritic().to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MSELoss = torch.nn.MSELoss()
    
    
    
    def convert_state(self, board):
        piece_bitboards = {}
        # for each color (white, black)
        for color in chess.COLORS:
            # for each piece type (pawn, bishop, knigh, rook, queen, king)
            for piece_type in chess.PIECE_TYPES:
                v = board.pieces_mask(piece_type, color) # vd: white PAWN -> 0000000000000000000000000000000000000000000000001111111100000000 = 65280
                symbol = chess.piece_symbol(piece_type)
                i = symbol.upper() if color else symbol # nếu màu trắng thì in hoa còn không in thường vd: white pawn -> P
                piece_bitboards[i] = v 
        # empty bitboard
        piece_bitboards['-'] = board.occupied ^ 2 ** 64 -1 # xor
        
        # player bitboard (full 1s if player is white, full 0s otherwise)
        player = 2 ** 64 - 1 if board.turn else 0
        
        # nhập thành
        castling_rights = board.castling_rights
        
        # tốt qua đường
        en_passant = 0 # vị trí của ô bắt tốt qua đường
        ep = board.ep_square
        
        if ep is not None:
            en_passant |= (1 << ep)
            
        bitboards = [b for b in piece_bitboards.values()] + [player] + [castling_rights] + [en_passant]
        
        bitarray = np.array([
            np.array([(bitboard >> i & 1) for i in range(64)]) for bitboard in bitboards
        ]).reshape((16, 8, 8))
        
        return bitarray
    
    def get_move_index(self, move):
        return 64 * (move.from_square) + (move.to_square)
    
    def mask_legal_move(self, board):
        mask = np.zeros((64,64))
        valid_moves_dict = {}
        
        for move in board.legal_moves:
            mask[move.from_square, move.to_square] = 1
            valid_moves_dict[self.get_move_index(move)] = move
    
        return torch.from_numpy(mask.flatten()).to(device), valid_moves_dict
    
    def select_action(self, board):
        bit_state = self.convert_state(board)
        
        # get valid moves
        mask, valid_moves_dict = self.mask_legal_move(board)
        
        with torch.no_grad():
            curr_state = torch.from_numpy(bit_state).float().unsqueeze(0).to(device) # (1,16,8,8)
            action, action_log_prob, state_val = self.policy_old.action(state=curr_state, mask=mask)
            
            self.buffer.states.append(curr_state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_log_prob)
            self.buffer.state_values.append(state_val)
            
            action = action.item()
            chosen_move = valid_moves_dict[action]
        
            return action, chosen_move
    
    def update(self):
        # Ước tính lợi nhuận của Monte Carlo cho lợi nhuẩn giảm chiết
        dis_rewards = []
        discounted_reward = 0
        
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminate)):
            if is_terminal:
                discounted_reward = 0
            
            discounted_reward = reward + (self.gamma * discounted_reward)
            dis_rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards
        dis_rewards = torch.tensor(dis_rewards, dtype=torch.float32)
        dis_rewards = (dis_rewards - dis_rewards.mean()) / (dis_rewards.std() + 1e-7)
        #===> dis_rewards: Có kích thước (N,), với N là số lượng phần tử trong danh sách


        # Convert list to Tensor
        # Ban đầu là 1 danh sách chứa các giá trị tensor, sử dụng stack để xếp chồng các giá trị tensor đó thành 1 giá trị tensor duy nhất (1 rows)
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)
        
        
        # ==> shape: Có kích thước (N, shape_element), với N là số lượng phần tử trong danh sách
        
        # calculate advantage
        # Lợi thế đo lường sự khác biệt giữa giá trị dự đoán của trạng thái và lợi nhuận thực tế thu được.
        # Lợi thế lớn: Điều này có thể cho thấy rằng hành động hiện tại đang mang lại kết quả tốt hơn nhiều so 
        #     với dự đoán trước đó, và bạn muốn khuyến khích hành động đó để cải thiện chính sách.
        # Lợi thế nhỏ hoặc gần bằng 0: Điều này cho thấy chính sách hiện tại đã dự đoán giá trị của trạng thái 
        #     khá chính xác. Do đó, hành động đó có thể không cần thay đổi nhiều. 
        advantages = dis_rewards.detach() - old_state_values.detach()
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values with new policy to calculate probability ratios
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratios (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach()) # về bản chất vẫn là Prob(a|s)/ Prob_old (a|s) [a: action, s:state]
            
            surr1 = ratios * advantages
        
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # final loss of clipped objective PPO
            # 0.01 * dist_entropy: khuyến khích sự đa dạng trong các hành động. (entropy cao -> đa dạng hành động và loss nhỏ )
            loss = -torch.min(surr1, surr2) + self.MSELoss(state_values, dis_rewards) - 0.01 * dist_entropy 
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
            # print(loss.mean().item())
        
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # clean buffer để chuẩn bị cho đợt huấn luyện tiếp
        self.buffer.clean()
        
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
        
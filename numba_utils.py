from numba import njit
import numpy as np

#avoid using strings w numba, better to assign ints  
ROLL = 0; PASS = 1; HOG_CALL_1 = 2; HOG_CALL_2 = 3
PIGGYBACK = -2; OINKER = -1; PIG_OUT= 0
REASON_NONE = 0; REASON_PIGGYBACK = 1; REASON_GOAL = 2; REASON_GOAL_HOG_CALL = 3




#LOGIC FROM STEP() IN GYM ENV
@njit 
def step_logic(players, player, action, throw_outcome, throw_score, WITH_HOG_CALLS, HOG_CALL_SCORE_1, HOG_CALL_SCORE_2, GOAL):
    points_this_action = 0
    next_player = player 
    opponent = (player+1) %2     #only for 2 players
    done = False; reward = 0 # Just for action NONE (only in interactive mode) 
    winner = -1
    reason_code = REASON_NONE     


    if action == PASS or action == HOG_CALL_1 or action == HOG_CALL_2:       #0,2,3
        points_this_action = players[player][2]        #TURN_SCORE
        players[player][0] += players[player][2]       #OWN_SCORE    
        players[opponent][1] = players[player][0]      #OPP_SCORE
        players[player][2] = 0                         #resets TURN_SCORE
        #   print(f"[DEBUG] Player {player} scores: own={players[player][0]}, turn={players[player][2]}")


        if WITH_HOG_CALLS: 
            players[player][3] = 0             #clears current player HOG_CALL
            players[opponent][3] = action-1    #stored on opponent, now they need to respond when rolling

        if players[player][0] >= GOAL:
            done = True
            # print("DONE - numba_utils")
            reason_code = REASON_GOAL 
            winner = player

        players[opponent][4] = action           #OPP_LAST_ACTION
        players[opponent][5] = points_this_action         #OPP_LAST_POINTS    
        next_player = opponent      

        return players, next_player, reward, done, points_this_action, reason_code, winner


    if action == ROLL:    #ROLL action, 1 
        score = throw_score    #outcome

        if score == 0:     #BAD roll outcome 
            if throw_outcome == OINKER or throw_outcome == PIGGYBACK: 
                reward = -players[player][0]
                points_this_action = -players[player][0]   
                players[player][0] = 0 
                if throw_outcome == PIGGYBACK: 
                    done = True; reason_code = REASON_PIGGYBACK
                    # print("DONE - numba_utils")
                    winner = opponent
            else:    #PIG_OUT
                reward = 0
                points_this_action = 0 

            players[opponent][1] = players[player][0]     #OPP_SCORE 
            players[player][2] = 0           #reset TURN_SCORE

            if WITH_HOG_CALLS:    #clears hog calls
                players[player][3] = 0
                players[opponent][3] = 0 

            next_player = opponent
            return players, next_player, reward, done, points_this_action, reason_code, winner
        

        if WITH_HOG_CALLS and players[player][3] > 0:       #GOOD roll and a hog call
            hog_call = players[player][3]
            correct = (hog_call == 1 and score == HOG_CALL_SCORE_1) or (hog_call == 2 and score == HOG_CALL_SCORE_2)
            if correct: 
                delta = 2*score
                if delta > players[player][0]:    #penalizs current player score
                    delta = players[player][0]

                reward = -delta + 1.0                        # and gives immediate reward
                points_this_action = -delta
                players[player][0] -= delta 

                if players[player][0] < 0:
                    players[player][0] = 0   
                players[opponent][0] += delta

                players[player][1] = players[opponent][0]     #transfers points
                players[opponent][1] = players[player][0]

                players[player][2] = 0             #resets TURN_SCORE
                players[player][3] = 0
                players[opponent][3] = 0    
                players[opponent][4] = action
                players[opponent][5] = points_this_action   
                next_player= opponent 
                 
                return players, next_player, reward, done, points_this_action, reason_code, winner
            
            else:   #incorrect hog call, opponent 
                reward = 2 * score - 1.0
                points_this_action = 2 * score

                players[opponent][0] -= 2 * score
                if players[opponent][0] < 0:
                    players[opponent][0] = 0
                players[player][1] = players[opponent][0]

                players[player][2] += 2 * score                  #TURN_SCORE
                if players[player][2] + players[player][0] > GOAL:
                    players[player][0] += players[player][2]
                    done = True
                    # print("DONE - numba_utils")
                    reason_code = REASON_GOAL_HOG_CALL
                    winner = player

                players[player][3] = 0
                players[opponent][3] = 0
                players[opponent][4] = action
                players[opponent][5] = points_this_action
                next_player = player  # leader continues after incorrect hog call
                return players, next_player, reward, done, points_this_action, reason_code, winner

                    

        else:     #good roll without hog call
            #  reward = score - 0.1*(GOAL - (players[player][0]+players[player][2]))     #made it more conservaive
            #  reward = points_this_action + 0.01 * players[player][0]
             reward = points_this_action + 0.01 * (players[player][0] - players[opponent][0])
             points_this_action = score
             players[player][2] += score

             if players[player][2] + players[player][0] > GOAL:
                players[player][0] += players[player][2]
                done = True
                # print("DONE - numba_utils")
                reason_code = REASON_GOAL
                winner = player

             return players, next_player, reward, done, points_this_action, reason_code, winner


    #default BUT SHOULD NOT HAPPEN - here as a fallback 
    print("DEBUG - default return in step_logic = bad")        
    return players, next_player, reward, done, points_this_action, reason_code, winner

        

              








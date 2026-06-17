from mjlab.tasks.standup.standup_env_cfg import StandUpEnvCfg

class G1StandUpGroundCfg(StandUpEnvCfg):
    class init_state(StandUpEnvCfg.init_state):
        pos = [0.0, 0.0, 0.5]
        rot = [0.7071, 0.0, -0.7071, 0.0]  # Pitch rotated -90 deg from Isaac [0, -1, 0, 1]
        no_orientation = False

class G1StandUpGroundProneCfg(StandUpEnvCfg):
    class init_state(StandUpEnvCfg.init_state):
        pos = [0.0, 0.0, 0.5]
        rot = [0.7071, 0.0, 0.7071, 0.0]   # Pitch rotated 90 deg from Isaac [0, 1, 0, 1]
        no_orientation = True

class G1StandUpPlatformCfg(StandUpEnvCfg):
    class init_state(StandUpEnvCfg.init_state):
        pos = [0.0, 0.0, 0.5]
        rot = [0.7071, 0.0, -0.7071, 0.0]
        no_orientation = False

class G1StandUpSlopeCfg(StandUpEnvCfg):
    class init_state(StandUpEnvCfg.init_state):
        pos = [0.0, 0.0, 0.8]
        rot = [0.7071, 0.0, -0.7071, 0.0]
        no_orientation = False

class G1StandUpWallCfg(StandUpEnvCfg):
    class init_state(StandUpEnvCfg.init_state):
        pos = [0.0, 0.0, 0.45]
        rot = [1.0, 0.0, 0.0, 0.0]
        no_orientation = False
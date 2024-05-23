from lbforaging.foraging.environment import ForagingEnv

def custom_spawn_food(self,max_food = 2,max_level = 3):
    min_level = max_level if self.force_coop else 1
    specific_locations = [(2, 2, 2), (4, 4, 1)] # Locations and levels of two apples

    for row,col,level in specific_locations:

        if (
                self.neighborhood(row, col).sum() > 0
                or self.neighborhood(row, col, distance=2, ignore_diag=True) > 0
                or not self._is_empty_location(row, col)
            ):
                continue
        self.field[row, col] = level
    self._food_spawned = self.field.sum()


def custom_spawn_players(self, max_player_level=1):
        for player in self.players:

            attempts = 0
            player.reward = 0

            while attempts < 1000:
                row = self.np_random.randint(0, self.rows)
                col = self.np_random.randint(0, self.cols)
                if self._is_empty_location(row, col):
                    player.setup(
                        (row, col),
                        1,
                        self.field_size,
                    )
                    break
                attempts += 1
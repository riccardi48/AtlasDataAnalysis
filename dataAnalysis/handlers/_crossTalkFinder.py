from dataAnalysis._types import clusterClass, dataAnalysis, clusterArray
from dataAnalysis._dependencies import (
    np,  # numpy
    npt,  # numpy.typing
    njit,  # numba
    numba,  # numba
)


class crossTalkFinder:
    def __init__(self) -> None:
        # Precompute the cross-talk dictionary once
        raw_dict = self._build_crosstalk_dict()
        # Convert to numba-compatible format for JIT compilation
        self.crossTalkDict = self._convert_to_numba_dict(raw_dict)

    def findCrossTalk_OneCluster(self, cluster) -> npt.NDArray[np.bool_]:
        shortIndexes = cluster.getShortIndexes()
        ToTs = cluster.getToTs()
        columns = cluster.getColumns()
        rows = cluster.getRows()

        # Use JIT-compiled function for the heavy lifting
        return self._find_crosstalk_jit(
            np.ascontiguousarray(ToTs, dtype=np.int32),
            np.ascontiguousarray(columns, dtype=np.int32),
            np.ascontiguousarray(rows, dtype=np.int32),
            self.crossTalkDict,
        )

    @staticmethod
    @njit(cache=True)
    def _find_crosstalk_jit(ToTs, columns, rows, crossTalkDict):
        n_pixels = len(ToTs)
        crossTalk = np.zeros(n_pixels, dtype=numba.boolean)

        for pixel_idx in range(n_pixels):
            if crossTalk[pixel_idx]:
                continue

            pixel_row = rows[pixel_idx]
            pixel_col = columns[pixel_idx]
            pixel_tot = ToTs[pixel_idx]

            # Skip if no cross-talk expected for this row
            if pixel_row not in crossTalkDict:
                continue

            expected_crosstalk = crossTalkDict[pixel_row]
            if len(expected_crosstalk) == 0:
                continue

            # Check each expected cross-talk row
            for cross_talk_idx in range(len(expected_crosstalk)):
                cross_row = expected_crosstalk[cross_talk_idx][0]

                # Find clash pixels: same column, target row, not already marked
                for clash_idx in range(n_pixels):
                    if (
                        crossTalk[clash_idx]
                        or clash_idx == pixel_idx
                        or columns[clash_idx] != pixel_col
                        or rows[clash_idx] != cross_row
                    ):
                        continue

                    clash_tot = ToTs[clash_idx]

                    # Apply cross-talk detection logic
                    if not (pixel_tot < 30 and clash_tot < 30):
                        if (clash_tot >= 255 and 30 <= pixel_tot < 255) or (
                            pixel_tot >= clash_tot and pixel_tot < 255 and clash_tot < 30
                        ):
                            crossTalk[clash_idx] = True

        return crossTalk

    def _convert_to_numba_dict(self, raw_dict):
        """Convert Python dict to numba-compatible typed dict"""
        # Create numba typed dict
        nb_dict = numba.typed.Dict.empty(key_type=numba.int32, value_type=numba.int32[:, :])

        for key, value in raw_dict.items():
            if value.size > 0:
                nb_dict[key] = value.astype(np.int32)
            else:
                nb_dict[key] = np.empty((0, 2), dtype=np.int32)

        return nb_dict

    def _build_crosstalk_dict(self) -> dict[int, npt.NDArray[np.int_]]:
        """Build cross-talk dictionary more efficiently"""
        crossTalkArray = self.calcCrossTalkArray()
        crossTalkDict = {}

        # Pre-allocate lists for better performance
        for i in range(372):
            pairs = []

            # Find all occurrences of i in the array
            rows_with_i, cols_with_i = np.where(crossTalkArray == i)

            for row_idx in rows_with_i:
                row_data = crossTalkArray[row_idx]
                for val in row_data:
                    if val != -1 and not (i - 2 <= val <= i + 2):
                        pairs.append([val, i])

            crossTalkDict[i] = (
                np.array(pairs, dtype=np.int32) if pairs else np.empty((0, 2), dtype=np.int32)
            )

        return crossTalkDict

    def calcCrossTalkArray(self) -> npt.NDArray[np.int_]:
        """Optimized array calculation with vectorization where possible"""
        up_to = 124
        crossTalkArray = np.full((up_to, 9), -1, dtype=np.int32)
        crossTalkArray[:, 0] = np.arange(0, up_to)

        # Vectorize some of the repetitive calculations
        rows = np.arange(1, up_to)

        # Process ranges more efficiently
        for row in rows:
            col_idx = 1

            # Group conditions and batch process
            if row <= 18:
                vals = [248 - row + 17, 248 - row + 18, 248 - row + 19]
                crossTalkArray[row, col_idx : col_idx + 3] = vals
                col_idx += 3

            if 18 <= row <= 104:
                vals = [row + 247, row + 248, row + 249]
                crossTalkArray[row, col_idx : col_idx + 3] = vals
                col_idx += 3
            elif 105 <= row <= 123:
                vals = [104 - row + 372, 104 - row + 371]
                if row >= 106:
                    vals.append(104 - row + 373)
                crossTalkArray[row, col_idx : col_idx + len(vals)] = vals
                col_idx += len(vals)

            # Additional conditions
            if row <= 12:
                vals = [198 - row, 197 - row]
                crossTalkArray[row, col_idx : col_idx + 2] = vals
                col_idx += 2
            elif 13 <= row <= 51 or 54 <= row <= 61:
                vals = [row + 185, row + 186]
                crossTalkArray[row, col_idx : col_idx + 2] = vals
                col_idx += 2
            elif 62 <= row <= 104:
                vals = [row + 80, row + 81, row + 82]
                crossTalkArray[row, col_idx : col_idx + 3] = vals
                col_idx += 3
            elif 106 <= row <= 122:
                crossTalkArray[row, col_idx] = row + 19
            elif row == 123:
                crossTalkArray[row, col_idx] = row + 62

        # Vectorized cleanup
        crossTalkArray[crossTalkArray > 372] = -1

        # Add final row efficiently
        final_row = np.full((1, 9), -1, dtype=np.int32)
        final_row[0, :2] = [248, 267]

        return np.vstack([crossTalkArray, final_row])

    # Fallback method without numba (if numba not available)
    def findCrossTalk_OneCluster_fallback(self, cluster) -> npt.NDArray[np.bool_]:
        shortIndexes = cluster.getShortIndexes()
        n_pixels = len(shortIndexes)
        crossTalk = np.zeros(n_pixels, dtype=bool)
        ToTs = cluster.getToTs()
        columns = cluster.getColumns()
        rows = cluster.getRows()

        # Create column-to-indices mapping for O(1) lookup
        col_to_indices = {}
        for idx, col in enumerate(columns):
            if col not in col_to_indices:
                col_to_indices[col] = []
            col_to_indices[col].append(idx)

        for pixel_idx in range(n_pixels):
            if crossTalk[pixel_idx]:
                continue

            pixel_row = rows[pixel_idx]
            pixel_col = columns[pixel_idx]
            pixel_tot = ToTs[pixel_idx]

            # Get expected cross-talk (from precomputed dict)
            expected_crosstalk = self.crossTalkDict.get(pixel_row)
            if expected_crosstalk is None or len(expected_crosstalk) == 0:
                continue

            # Only check pixels in the same column
            same_col_indices = col_to_indices[pixel_col]

            for cross_row, _ in expected_crosstalk:
                for clash_idx in same_col_indices:
                    if (
                        crossTalk[clash_idx]
                        or clash_idx == pixel_idx
                        or rows[clash_idx] != cross_row
                    ):
                        continue

                    clash_tot = ToTs[clash_idx]

                    # Optimized conditions
                    if not (pixel_tot < 30 and clash_tot < 30):
                        if (clash_tot >= 255 and 30 <= pixel_tot < 255) or (
                            pixel_tot >= clash_tot and pixel_tot < 255 and clash_tot < 30
                        ):
                            crossTalk[clash_idx] = True

        return crossTalk

    # Keep original interface
    def crossTalkFunction(self) -> dict[int, npt.NDArray[np.int_]]:
        # Convert back to original format if needed
        result = {}
        for key in self.crossTalkDict:
            result[key] = np.array(self.crossTalkDict[key])
        return result

# All 51 categories used in the Washington RGB-D dataset
cats = ["apple", "cell_phone", "hand_towel", "notebook", "scissors", "food_bag", "cereal_box", "instant_noodles",
        "onion", "shampoo", "banana", "coffee_mug", "food_jar", "orange", "soda_can", "bell_pepper", "comb",
        "keyboard", "peach", "sponge", "bowl", "food_cup", "kleenex", "pear", "stapler", "food_box", "dry_battery",
        "lemon", "pitcher", "tomato", "calculator", "flashlight", "lightbulb", "plate", "toothbrush", "camera",
        "garlic", "lime", "pliers", "toothpaste", "food_can", "glue_stick", "marker", "potato", "water_bottle",
        "cap", "greens", "mushroom", "rubber_eraser", "ball", "binder"]

mRMR_selected_image_features = [3554, 0, 2, 2647, 5, 1897, 11, 2295, 3205, 3716, 12, 582, 3515, 280, 2343, 13, 1530, 15,
                                114, 16, 1571, 18, 19, 1674, 20, 3660, 22, 2744, 25, 3455, 26, 1130, 29, 31, 111, 33,
                                1090, 34, 3800, 36, 3014, 38, 1179, 1599, 39, 2882, 41, 3795, 42, 709, 55, 2674, 3869,
                                2470, 57, 1447, 59, 61, 1448, 62, 1175, 64, 1676, 65, 2865, 66, 71, 2905, 73, 1742, 74,
                                1853, 80, 2307, 83, 2990, 84, 588, 85, 86, 2521, 87, 2761, 91, 3889, 92, 2546, 93, 96,
                                737, 101, 108, 102, 1320, 103, 3213, 107, 109, 3159, 110, 2292, 116, 2459, 120, 1226,
                                121, 123, 2617, 124, 2573, 125, 2904, 128, 133, 3748, 135, 3761, 136, 3497, 137, 76,
                                143, 149, 3725, 156, 485, 157, 3071, 158, 2590, 160, 162, 478, 169, 1437, 171, 300, 172,
                                2254, 174, 181, 2556, 188, 1312, 192, 191, 195, 95, 196, 200, 538, 202, 2047, 203, 2938,
                                207, 3863, 208, 213, 145, 215, 1385, 216, 2875, 218, 3030, 220, 222, 1621, 233, 3356,
                                234, 3262, 237, 2532, 238, 2205, 240, 3551, 243, 245, 1669, 247, 3377, 250, 2104, 251,
                                733, 252, 253, 2059, 255, 2661, 257, 3169, 259, 260, 2387, 261, 2163, 262, 3162, 265,
                                2440, 266, 273, 298, 274, 2870, 277, 2715, 278, 1964, 279, 282, 1562, 284, 2278, 286,
                                3110, 287, 4065, 288, 290, 3957, 291, 1799, 294, 1183, 296, 1144, 301, 304, 3790, 305,
                                1733, 306, 37, 312, 2644, 315, 316, 743, 322, 2316, 324, 3272, 330, 3191, 333, 341,
                                3668, 347, 715, 350, 1907, 351, 352, 2391, 353, 4076, 354, 3107, 355, 508, 356, 357,
                                1238, 358, 139, 359, 1063, 360, 946, 373, 382, 2740, 384, 1129, 385, 1049, 386, 2326,
                                388, 391, 3648, 396, 1859, 397, 3556, 398, 1726, 402, 403, 2089, 404, 3940, 406, 2757,
                                408, 3717, 411, 415, 3750, 416, 2131, 417, 2411, 418, 1442, 421, 422, 1194, 425, 3428,
                                427, 4071, 429, 431, 454, 434, 1128, 441, 148, 442, 1306, 443, 447, 3987, 449, 1480,
                                450, 923, 451, 3632, 453, 455, 1325, 457, 2629, 458, 3898, 464, 1771, 465, 468, 3128,
                                469, 49, 470, 3336, 472, 3016, 475, 477, 2019, 479, 2567, 484, 3385, 488, 555, 489, 490,
                                3990, 496, 4046, 500, 1844, 502, 506, 1910, 509, 400, 513, 3981, 516, 992, 523, 524,
                                3494, 525, 2310, 531, 426, 533, 3032, 535, 536, 2141, 537, 2786, 539, 1814, 540, 3354,
                                541, 542, 824, 544, 1794, 545, 3509, 551, 1180, 552, 556, 3967, 560, 2011, 564, 852,
                                565, 2892, 567, 568, 146, 569, 2820, 579, 2549, 583, 1786, 585, 586, 2670, 590, 1716,
                                591, 2837, 593, 594, 2484, 596, 9, 597, 964, 598, 1718, 602, 603, 3408, 604, 1586, 605,
                                2219, 606, 2511, 607, 609, 268, 610, 2073, 612, 3167, 613, 113, 615, 616, 1006, 617,
                                906, 619, 2114, 622, 2427, 625, 626, 1634, 628, 1547, 629, 2084, 630, 689, 631, 632,
                                474, 633, 4008, 634, 289, 637, 1698, 638, 640, 2357, 643, 1429, 644, 2395, 645, 647,
                                2142, 649, 117, 651, 1891, 654, 878, 655, 656, 3387, 658, 2126, 659, 409, 661, 1650,
                                663, 666, 3783]

def get_pairs(scene, rate, uid, unique_source_view=False):
    l = len(scene["images"])

    try:
        assert set(range(l)) == set(scene["images"]) == set(scene["cams"]), (
            l,
            set(scene["images"]),
            set(scene["cams"]),
        )
    except AssertionError as e:
        print("corruped scene!")
        print(l)
        print(set(scene["images"]))
        print(set(scene["cams"]))
        return

    cams = np.array([scene["cams"][i] for i in range(l)])

    # import pdb
    # pdb.set_trace()

    if unique_source_view:
        # print(np.array(list(scene['images'])))
        n_samples = int(rate * l) + 1
        # print(n_samples)
        # idxs = np.random.choice(
        #     np.array(list(scene["images"])), n_samples, replace=False
        # )
        idxs = range(1, 17)
        pair_idxs = [(idx, 0) for idx in idxs]
    else:
        pair_idxs = [
            np.random.choice(np.array(list(scene["images"])), (2,), replace=False)
            for _ in range(int(rate * l))
        ]

    for i, j in pair_idxs:
        pair_uid = "%.4d__%.4d__%s" % (i, j, uid)
        # print(pair_uid)
        # print(scene['images'].keys())
        yield scene["images"][i], scene["images"][j], i, j, uid, pair_uid, cams

def objaverse_compose_fn(samples, rate):
    # rate*l pairs will be sampled from a scene of length l
    # lower rates have more unbiased sampling but fewer imgs/s
    cur_uid = None
    uid_ctr = 0
    scene = get_new_scene()

    for sample in samples:
        # if "png" in sample:
        #     sample_type = "png"
        # elif "npy" in sample:
        #     sample_type = "npy"
        # else:
        #     raise NotImplementedError
        # assert ('png' in sample) ^ ('npy' in sample), sample.keys()

        # import pdb
        # pdb.set_trace()

        *_, uid, str_idx = sample["__key__"].split("/")
        # print(uid, str_idx)

        # import pdb
        # pdb.set_trace()

        if cur_uid != uid:
            # print(uid_ctr)
            uid_ctr += 1

            if uid_ctr % 10_000 == 0:
                print(f"Processing scene: {uid_ctr}!")

            yield from get_pairs(scene, rate, uid)

            scene = get_new_scene()
            cur_uid = uid

        idx = int(str_idx)

        if "png" in sample:
            scene["images"][idx] = sample["png"]
        if "npy" in sample:
            with io.BytesIO(sample["npy"]) as fp:
                cams = np.load(fp)
            # cams = np.zeros(shape=(80, 3, 4))
            scene["cams"][idx] = cams

        if "npy" not in sample and "png" not in sample:
            raise NotImplementedError


def get_image_from_bytes_objaverse(image_bytes):
    with io.BytesIO(image_bytes) as fp:
        img = imageio.imread(fp)
    img = img / 255.0
    img[img[:, :, -1] == 0.0] = [1.0, 1.0, 1.0, 1.0]
    img = cv2.resize(img[..., :3], (256, 256))
    img = img * 2 - 1
    return img
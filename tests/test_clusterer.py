from sentistream.worker.clusterer import StreamClusterer


def test_get_cluster_returns_int_and_increments():
    clusterer = StreamClusterer(n_dimensions=5)
    cluster_id = clusterer.get_cluster([0.1, 0.2, 0.3, 0.4, 0.5])

    assert isinstance(cluster_id, int)
    assert clusterer.records_processed == 1


def test_active_clusters_info_returns_list():
    clusterer = StreamClusterer(n_dimensions=5)
    info = clusterer.get_active_clusters_info()

    assert isinstance(info, list)

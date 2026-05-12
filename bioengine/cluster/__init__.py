"""``bioengine.cluster`` is intentionally a namespace-only package.

We do NOT eagerly import ``RayCluster`` here because doing so pulls in
``bioengine.utils`` (and from there ``hypha_rpc``) — which would force
external Ray clusters (e.g. KubeRay deployments) to have BioEngine's
Hypha runtime deps installed just to deserialise an actor class shipped
via ``runtime_env.py_modules``.

Import submodules explicitly:

    from bioengine.cluster.ray_cluster import RayCluster
    from bioengine.cluster.proxy_actor import BioEngineProxyActor
"""

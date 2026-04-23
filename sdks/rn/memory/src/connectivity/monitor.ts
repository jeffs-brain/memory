export type ConnectivitySnapshot = {
  readonly online: boolean
  readonly reachable: boolean | null
  readonly changedAt: Date
}

export type ConnectivityListener = (snapshot: ConnectivitySnapshot) => void

export type ConnectivityAdapter = {
  current(): Promise<ConnectivitySnapshot>
  subscribe(listener: ConnectivityListener): Promise<() => void> | (() => void)
}

export type ConnectivityMonitor = {
  snapshot(): ConnectivitySnapshot
  refresh(): Promise<ConnectivitySnapshot>
  subscribe(listener: ConnectivityListener): () => void
  close(): Promise<void>
}

const DEFAULT_SNAPSHOT: ConnectivitySnapshot = {
  online: true,
  reachable: null,
  changedAt: new Date(),
}

const buildNetInfoAdapter = async (): Promise<ConnectivityAdapter> => {
  const netInfo = await import('@react-native-community/netinfo')
  const currentState = async (): Promise<ConnectivitySnapshot> => {
    const snapshot = await netInfo.fetch()
    return {
      online: snapshot.isConnected === true,
      reachable: snapshot.isInternetReachable,
      changedAt: new Date(),
    }
  }
  return {
    current: currentState,
    subscribe: (listener) =>
      netInfo.addEventListener((state) => {
        listener({
          online: state.isConnected === true,
          reachable: state.isInternetReachable,
          changedAt: new Date(),
        })
      }),
  }
}

export const createConnectivityMonitor = async (
  options: {
    readonly adapter?: ConnectivityAdapter
    readonly initialSnapshot?: ConnectivitySnapshot
  } = {},
): Promise<ConnectivityMonitor> => {
  const adapter =
    options.adapter ??
    (await buildNetInfoAdapter().catch(
      () =>
        ({
          current: async () => DEFAULT_SNAPSHOT,
          subscribe: () => () => {},
        }) as ConnectivityAdapter,
    ))

  let current = options.initialSnapshot ?? (await adapter.current())
  const listeners = new Set<ConnectivityListener>()
  const unsubscribe = await adapter.subscribe((snapshot) => {
    current = snapshot
    for (const listener of listeners) listener(snapshot)
  })

  return {
    snapshot: () => current,
    refresh: async () => {
      current = await adapter.current()
      return current
    },
    subscribe: (listener) => {
      listeners.add(listener)
      return () => {
        listeners.delete(listener)
      }
    },
    close: async () => {
      listeners.clear()
      unsubscribe()
    },
  }
}

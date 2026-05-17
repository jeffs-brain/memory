// SPDX-License-Identifier: Apache-2.0

/**
 * Connector registry for registering and looking up connector factories
 * by name.
 */

import type { ConnectorConfig, Connector, ConnectorFactory } from './types.js'

/** Returned when a registry lookup finds no connector with the given name. */
export class ConnectorNotFoundError extends Error {
  constructor(name: string) {
    super(`Connector not found: ${name}`)
    this.name = 'ConnectorNotFoundError'
  }
}

/** Returned when registering a connector with a name already taken. */
export class ConnectorExistsError extends Error {
  constructor(name: string) {
    super(`Connector already registered: ${name}`)
    this.name = 'ConnectorExistsError'
  }
}

/**
 * Registry provides registration and lookup of connector factories by
 * name.
 */
export class ConnectorRegistry {
  private readonly factories = new Map<string, ConnectorFactory>()

  /**
   * Register a connector factory under the given name. Throws
   * ConnectorExistsError if the name is already taken.
   */
  register(name: string, factory: ConnectorFactory): void {
    if (this.factories.has(name)) {
      throw new ConnectorExistsError(name)
    }
    this.factories.set(name, factory)
  }

  /**
   * Create a new Connector instance for the given name using the
   * registered factory and provided configuration. Throws
   * ConnectorNotFoundError if no factory is registered.
   */
  lookup(name: string, cfg: ConnectorConfig): Connector {
    const factory = this.factories.get(name)
    if (!factory) {
      throw new ConnectorNotFoundError(name)
    }
    return factory({ ...cfg, name })
  }

  /** Return all registered connector names. */
  names(): readonly string[] {
    return [...this.factories.keys()]
  }

  /** Check whether a connector factory is registered under the name. */
  has(name: string): boolean {
    return this.factories.has(name)
  }
}
